"""Foundry Local üzerinden yerel LLM çağrıları.

Foundry Local, OpenAI-uyumlu bir local endpoint sağlar (bkz. foundry-local-sdk).
Bu modül, Foundry Local kurulu değilse (örn. geliştirme ortamı) hata fırlatmak yerine
açıklayıcı bir istisna verir; böylece üst katmanlar kullanıcıya net bilgi gösterebilir.
"""
from __future__ import annotations

import importlib
import logging
import re
import subprocess
import time
from functools import lru_cache

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


def _collect_transport_error_types() -> tuple[type[BaseException], ...]:
    """Akış sırasında "bağlantı koptu" anlamına gelen istisna sınıfları.

    NEDEN BİR LİSTE: `openai` istemcisi sürüme göre FARKLI bir HTTP kütüphanesi
    kullanıyor — eski sürümler `httpx`, 3.x ise `httpx2`. Bu iki paketin
    istisna sınıfları birbirinden tamamen bağımsız: `httpx2.RemoteProtocolError`,
    `httpx.TransportError`'ın alt sınıfı DEĞİL. Kod yalnızca `httpx`i
    yakaladığı için, `httpx2` kullanan bir kurulumda Foundry Local'in bilinen
    "üretim biter bitmez bağlantıyı yarıda kesme" davranışı (bkz.
    `_stream_and_strip`) hiç yakalanmıyor ve kullanıcıya ham traceback olarak
    çıkıyordu (gözlemlenen: `/quiz/stream` çöküyor).

    Hangisi kuruluysa ikisini de yakalıyoruz; hiçbiri yoksa `OSError`'a
    düşüyoruz (yakalanamayan bir şeye göre daha güvenli bir taban).
    """
    types: list[type[BaseException]] = []
    for module_name in ("httpx", "httpx2"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        error_type = getattr(module, "TransportError", None)
        if isinstance(error_type, type) and issubclass(error_type, BaseException):
            types.append(error_type)
    return tuple(types) or (OSError,)


TRANSPORT_ERRORS = _collect_transport_error_types()


# CUDA bağlamının bozulduğunu (kurtarılamaz) gösteren imzalar.
_FATAL_GPU_MARKERS = (
    "illegal memory access",
    "device-side assert",
    "unspecified launch failure",
)

# Geçici bellek darlığını (daha küçük bir istekle çalışabilir) gösteren
# imzalar. ÖNEMLİ: "CudaMallocArray" her iki hatada da geçtiği için burada
# YOK — önce ölümcül kontrol yapılıyor (bkz. `is_capacity_error`).
_CAPACITY_MARKERS = ("out of memory", "outofmemory", "oom")

_GPU_CONTEXT_LOST_MESSAGE = (
    "GPU'nun CUDA bağlamı bozuldu (illegal memory access). Bu, model kartın "
    "belleğine sığmadığında oluşuyor ve backend süreci içinde düzeltilemiyor: "
    "Foundry Local servisini yeniden başlatman gerekiyor "
    "(scripts/foundry-doctor.ps1). Tekrarlıyorsa .env'de daha küçük bir "
    "model seç (FOUNDRY_MODEL_ALIAS)."
)


def is_fatal_gpu_error(exc: Exception) -> bool:
    """CUDA bağlamı bozuldu mu? (bkz. `GpuContextLost`)"""
    return any(marker in str(exc).lower() for marker in _FATAL_GPU_MARKERS)


def is_capacity_error(exc: Exception) -> bool:
    """Geçici bellek darlığı mı? Ölümcül bağlam kaybıyla KARIŞTIRILMAMALI:
    bunda daha küçük bir istekle tekrar denemek işe yarıyor, diğerinde
    yaramıyor."""
    if is_fatal_gpu_error(exc):
        return False
    return any(marker in str(exc).lower() for marker in _CAPACITY_MARKERS)


def raise_if_gpu_context_lost(exc: Exception) -> None:
    """Hata bağlam kaybıysa, kullanıcıya gösterilebilir bir `GpuContextLost`a
    çevirip fırlatır; değilse hiçbir şey yapmaz."""
    if is_fatal_gpu_error(exc):
        raise GpuContextLost(_GPU_CONTEXT_LOST_MESSAGE) from exc


class FoundryNotAvailable(RuntimeError):
    """Foundry Local servisine erişilemediğinde fırlatılır."""


class DegenerateOutput(RuntimeError):
    """Model bozuk bir tekrar döngüsüne girip aynı kısa kelimeyi/parçayı
    defalarca art arda ürettiğinde fırlatılır (bkz. `_RepetitionGuard`)."""


class GpuContextLost(RuntimeError):
    """GPU'nun CUDA bağlamı bozulduğunda fırlatılır — süreç içinde
    KURTARILAMAZ bir durum.

    Gözlemlenen: Foundry Local, "CUDA error in CudaMallocArray ... an illegal
    memory access was encountered" mesajıyla 500 dönüyor. Bu hatanın kritik
    özelliği, ilk oluştuktan SONRA aynı CUDA bağlamındaki HER isteğin de
    başarısız olması: bağlam zehirlenmiş oluyor.

    Bunu doğrulayan gerçek gözlem: hata sonrası context 3145 -> 1562 -> 770
    -> 385 karaktere kadar küçültülerek tekrar denendi, dördü de aynı hatayı
    verdi. 385 karakterlik bir istek hiçbir GPU'yu zorlamayacağına göre sorun
    context boyutu değil, bozulmuş bağlamın kendisi.

    Bu yüzden bunu `_is_capacity_error`'dan (bölerek yeniden denemenin işe
    YARADIĞI, geçici bellek darlığı) ayrı tutuyoruz: burada tek çözüm Foundry
    Local servisini yeniden başlatmak, kalıcı çözüm ise karta sığan bir model
    kullanmak.
    """


class EmptyAnswer(RuntimeError):
    """Model bir yanıt tamamladı ama kullanıcıya gösterilecek görünür metin
    boş (ya da anlamsız derecede kısa, bkz. `_looks_too_short`) çıktı.
    Gözlemlenen bir Qwen3 arıza modu: `/no_think` soft-switch'i (bkz.
    `build_prompt`) akıl yürütmeyi her zaman tam olarak bastırmıyor; model
    `<think>...</think>` içindeyken `max_tokens` sınırına ulaşırsa,
    `_ThinkStreamStripper`/`strip_think` (kasıtlı olarak) yarım kalmış akıl
    yürütme metnini kullanıcıya göstermek yerine düşürüyor — sonuçta ya
    hiçbir şey yayınlanmıyor ya da geriye tek bir noktalama işareti gibi
    (`"."`) anlamsız bir kırıntı kalıyor (gözlemlenen bir kullanıcı ekran
    görüntüsü). `generate_answer`/`generate_answer_stream` bunu sessiz bir
    boş/anlamsız cevap yerine, giderek daha geniş bir token bütçesiyle
    otomatik olarak yeniden denedikten sonra (bkz. `_answer_token_budgets`)
    hâlâ başarısızsa kullanıcının ne olduğunu anlayabileceği açık bir
    hataya çeviriyor."""


def _patch_cli_compat() -> None:
    """foundry-local-sdk 0.5.1, servis yönetimi için `foundry service ...`
    komutunu çağırıyor. Foundry Local CLI 0.10.3'te bu alt komut `foundry
    server ...` olarak değiştirildi (`foundry service status` ->
    "Unknown command: 'service'"). SDK içindeki `foundry_local.api` modülü
    `get_service_uri`/`start_service` isimlerini kendi global namespace'ine
    import-time'da bağladığı için yamayı orada uygulamamız gerekiyor
    (foundry_local.service üzerinde patch yapmak api.py'yi etkilemez).

    site-packages'e dokunmuyoruz; bu yalnızca process içi, çalışma zamanı
    bir düzeltme.
    """
    import foundry_local.api as _foundry_api

    if getattr(_foundry_api, "_ask_me_cli_patched", False):
        return  # zaten yamalı

    def _get_service_uri_text() -> str | None:
        with subprocess.Popen(
            ["foundry", "server", "status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as proc:
            stdout, _ = proc.communicate()
            text = stdout.decode("utf-8", errors="replace")
            match = re.search(r"http://(?:[a-zA-Z0-9.-]+|\d{1,3}(\.\d{1,3}){3}):\d+", text)
            return match.group(0) if match else None

    def _http_ready(uri: str) -> bool:
        """`foundry server status` çıktısında bir URI görünmesi, o adreste
        HTTP dinleyicisinin gerçekten bağlantı kabul ettiği anlamına gelmiyor.
        Soğuk başlangıçta CLI, dinleyici soket'e bağlanmadan hemen önce bile
        bu URI'yi metne yazabiliyor; bu da servis "hazır" sanılıp hemen
        `/v1/models` çağrıldığında "connection actively refused" (WinError
        10061) hatasına yol açan yarış durumuydu. Bu yüzden metni bulduktan
        sonra ayrıca gerçek bir HTTP isteğiyle doğruluyoruz.
        """
        import httpx

        try:
            httpx.get(f"{uri}/v1/models", timeout=2)
            return True
        except httpx.TransportError:
            return False

    def _get_service_uri() -> str | None:
        uri = _get_service_uri_text()
        if uri is not None and _http_ready(uri):
            return uri
        return None

    def _start_service() -> str | None:
        if (uri := _get_service_uri()) is not None:
            return uri
        # Soğuk başlangıçta (CUDA/WebGPU/OpenVINO/TensorRT sağlayıcı kaydı)
        # servis 5-10 saniyeden çok daha uzun sürebiliyor; erken vazgeçip
        # devam etmek, `foundry model load`'ın servisi ayakta bulamayıp
        # KENDİ başlatma denemesini yapmasına ve "spawn lock" çakışmasına
        # yol açıyordu. Bu yüzden burada gerçekten hazır olana kadar (en
        # fazla 90 saniye) bekliyoruz — ve "hazır" demeden önce hem CLI
        # metnini hem de gerçek HTTP bağlantısını (_get_service_uri
        # üzerinden) doğruluyoruz.
        with subprocess.Popen(["foundry", "server", "start"]):
            for _ in range(90):
                if (uri := _get_service_uri()) is not None:
                    return uri
                time.sleep(1)
            return None

    _foundry_api.get_service_uri = _get_service_uri
    _foundry_api.start_service = _start_service
    _foundry_api._ask_me_cli_patched = True


@lru_cache(maxsize=1)
def _get_manager():
    """Foundry Local servisine bağlanır.

    NOT: SDK 0.5.1'in `download_model`/`load_model`/`get_model_info` metodları
    CLI 0.10.3'ün REST API'sinde artık var olmayan `/foundry/list` route'unu
    kullanıyor (404 döner). Bu yüzden manager'ı `bootstrap=False` ile açıp
    model yükleme ve model-id çözümlemesini kendimiz, çalıştığını doğruladığımız
    yollarla (CLI `model load` + `/v1/models`) yapıyoruz.
    """
    try:
        from foundry_local import FoundryLocalManager
    except ImportError as exc:  # pragma: no cover
        raise FoundryNotAvailable(
            "foundry-local-sdk kurulu değil. `pip install foundry-local-sdk` ile kurun "
            "ve Foundry Local uygulamasının çalıştığından emin olun."
        ) from exc

    _patch_cli_compat()

    try:
        manager = FoundryLocalManager(bootstrap=False)
        if not manager.is_service_running():
            manager.start_service()
        if not manager.is_service_running():
            raise FoundryNotAvailable(
                "Foundry Local servisi 90 saniye içinde ayağa kalkmadı. "
                "`foundry server status` / `foundry server logs` ile kontrol edin; "
                "takılı kalmış bir 'foundrylocald.exe' süreci varsa sonlandırıp tekrar deneyin."
            )
        return manager
    except Exception as exc:
        raise FoundryNotAvailable(
            "Foundry Local servisine bağlanılamadı. `foundry server start` ile "
            "servisi manuel başlatıp tekrar deneyin."
        ) from exc


def _ensure_model_loaded(alias: str) -> None:
    """Modelin bellekte yüklü olduğundan emin olur (CLI üzerinden).

    SDK'nın `load_model()` metodu da kırık katalog route'una bağımlı olduğu
    için burada doğrudan `foundry model load` komutunu kullanıyoruz — bu
    komutun çalıştığını manuel olarak doğruladık.

    NOT: `foundry model load`'ın kendi iç mantığı, servis bizim Python
    tarafımızdan (HTTP ile) doğrulanmış şekilde çalışıyor olsa bile bunu
    bazen göremeyip KENDİ ayrı bir daemon başlatma denemesi yapabiliyor;
    bu da kendi sabit ~15 saniyelik iç zaman aşımına takılıp
    "Daemon did not start listening within 15s" hatasıyla başarısız
    olabiliyor (gözlemlenen, geçici bir yarış durumu). Bu bizim
    `_start_service`'teki 90 saniyelik bekleyişimizin dışında, CLI'nin
    kendi kararı olduğu için Python tarafından önlenemiyor; bu yüzden
    birkaç kez tekrar deniyoruz.

    ÖNEMLİ: `capture_output=True` (stdout/stderr'i pipe'a yönlendirme)
    KASITLI OLARAK kullanılmıyor. Bu komut Python'dan `capture_output=True`
    ile (yani gerçek bir konsola/TTY'ye bağlı olmadan) çağrıldığında CLI
    tutarlı biçimde "Daemon did not start listening within 15s" hatası
    veriyor, ama TAM AYNI komut interaktif olarak (gerçek konsolda) her
    zaman sorunsuz çalışıyor — bu da CLI'nin pipe/non-TTY modunda farklı
    (ve bozuk) bir davranışa geçtiğini gösteriyor. Bu yüzden burada
    stdout/stderr'i pipe'a yönlendirmiyoruz; alt süreç backend'in kendi
    konsolunu miras alıyor, çıktısı doğrudan backend terminalinde görünür.
    """
    for attempt in range(3):
        result = subprocess.run(["foundry", "model", "load", alias])
        if result.returncode == 0:
            return
        if attempt < 2:
            time.sleep(3)

    raise FoundryNotAvailable(
        f"'{alias}' modeli 3 denemede de yüklenemedi. Backend'in çalıştığı terminaldeki "
        f"`foundry model load {alias}` çıktısına bakıp hatayı kontrol edin."
    )


@lru_cache(maxsize=1)
def _get_model_id() -> str:
    """`/v1/models` üzerinden alias'ın donanıma göre seçilmiş gerçek ID'sini bulur.

    Örn. alias "qwen3-8b" -> gerçek id "qwen3-8b-cuda-gpu" (GPU'lu makinede)
    ya da "qwen3-8b-cpu" (CPU-only makinede). Bunu sabit yazmak yerine
    dinamik çözmek, Foundry Local'ın otomatik donanım seçimiyle uyumlu kalır.
    """
    import httpx

    manager = _get_manager()
    alias = settings.foundry_model_alias

    # ÖNEMLİ: modelin `/v1/models` listesinde GÖRÜNMESİ, gerçekten belleğe
    # YÜKLENMİŞ olduğu anlamına gelmiyor (bunu burada denedik ve
    # "Model 'qwen3-8b-cuda-gpu' is not loaded" hatasıyla doğrulandı) —
    # daemon modeli sadece "bilinen/kataloglanmış" olarak listeleyebiliyor.
    # Bu yüzden `foundry model load`'ı HER ZAMAN çağırıyoruz; zaten bu
    # fonksiyon `lru_cache` ile sarmalı olduğu için başarılı olduktan sonra
    # backend süreci boyunca yeniden çalışmayacak (yani ekstra CLI çağrısı
    # maliyeti sadece ilk çağrıda oluşuyor).
    _ensure_model_loaded(alias)

    resp = httpx.get(f"{manager.service_uri}/v1/models", timeout=10)
    resp.raise_for_status()
    for model in resp.json().get("data", []):
        if model.get("parent") == alias or model.get("id") == alias:
            return model["id"]

    raise FoundryNotAvailable(
        f"'{alias}' modeli /v1/models listesinde bulunamadı. "
        f"`foundry model list` ile kontrol edin."
    )


# Sistem prompt'ları kasıtlı olarak KISA tutuldu: her istekte yeniden
# prefill edildikleri için uzun talimatlar hem gecikme hem de modelin asıl
# soruya odaklanmasını zorlaştıran gürültü demek. Eskiden burada "cevabın
# sonunda hangi kaynakları kullandığını yaz" talimatı vardı; kaldırıldı --
# kaynak listesi zaten arayüzde RAG sonucundan gösteriliyor, modelin bunu
# ikinci kez (ve çoğu zaman yanlış) üretmesi hem token hem doğruluk kaybıydı.
# Onun yerine cümle içinde [Kaynak N] atfı isteniyor.
SYSTEM_PROMPT_TR = (
    "Bilgisayar mühendisliği öğrencilerine yardımcı olan bir ders çalışma "
    "asistanısın. Sana verilen kaynak metinlerden yararlanarak Türkçe cevap ver. "
    "Kaynaklarda olmayan bilgiyi uydurma; emin değilsen bunu açıkça söyle. "
    "Bir bilgiyi hangi kaynaktan aldığını cümle içinde [Kaynak N] biçiminde belirt. "
    "Doğrudan cevaba gir: giriş cümlesi, soruyu tekrarlama ve gereksiz uzatma yok. "
    "Biçimlendirme için markdown kullan: cevap birkaç ayrı konuyu kapsıyorsa "
    "'## Başlık' ile bölümlere ayır, sıralanabilen şeyleri madde listesi yap, "
    "anahtar terimleri **kalın** yaz, kod ve kod parçalarını ``` bloğuna al. "
    "Kısa ve tek konulu bir cevapta başlık AÇMA — bir iki paragraf yeterli."
)

SYSTEM_PROMPT_EN = (
    "You are a study assistant for computer engineering students. "
    "Answer in English using the provided source excerpts. "
    "Do not invent information that is not in the sources; say so if you are unsure. "
    "Cite sources inline as [Kaynak N] where you use them. "
    "Get straight to the answer: no preamble, no restating the question, no padding. "
    "Use markdown for structure: split multi-topic answers into '## Heading' "
    "sections, use bullet lists for enumerable things, **bold** for key terms, "
    "and ``` blocks for code. Do NOT add headings to a short, single-topic "
    "answer — a paragraph or two is enough."
)

# Hybrid RAG yeterince ilgili hiçbir parça bulamadığında kullanılır (bkz.
# api/ask.py:_retrieve ve settings.min_relevance_score). Eski davranışta bu
# durumda da "yalnızca kaynakları kullan" talimatı gönderilip ALAKASIZ
# parçalar context'e konuyordu -- model ya o alakasız metinlerden zorlama bir
# cevap üretiyordu ya da "kaynaklarda yok" deyip kullanıcıyı elleri boş
# bırakıyordu. Artık genel bilgiyle cevap veriyor ama bunun dosyalardan
# gelmediğini açıkça söylüyor.
SYSTEM_PROMPT_NO_CONTEXT_TR = (
    "Bilgisayar mühendisliği öğrencilerine yardımcı olan bir ders çalışma "
    "asistanısın. Kullanıcının yüklediği dosyalarda bu soruyla ilgili bir bölüm "
    "bulunamadı. Soruyu kendi genel bilginle, Türkçe ve öz biçimde cevapla; "
    "cevabın başında bu bilginin yüklenen dosyalardan değil genel bilgiden "
    "geldiğini tek cümleyle belirt. Biçimlendirme için markdown kullan: uzun "
    "ve çok konulu cevaplarda '## Başlık' ve madde listeleri, anahtar "
    "terimlerde **kalın**, kodda ``` bloğu."
)

SYSTEM_PROMPT_NO_CONTEXT_EN = (
    "You are a study assistant for computer engineering students. "
    "No relevant section was found in the user's uploaded files for this question. "
    "Answer from your own general knowledge, concisely, in English; start with one "
    "sentence noting that this comes from general knowledge rather than the "
    "uploaded files. Use markdown for structure: '## Heading' sections and "
    "bullet lists for long multi-topic answers, **bold** for key terms, ``` "
    "blocks for code."
)

# Quiz üretimi (bkz. services/quiz.py) için ayrı sistem prompt'u. Normal
# cevap prompt'u "[Kaynak N] biçiminde atıf yap" diyor; quiz çıktısı ham JSON
# olarak parse edildiği için o talimat oraya sızarsa JSON'ı bozabiliyor.
SYSTEM_PROMPT_QUIZ_TR = (
    "Ders notlarından çalışma sorusu üreten bir asistansın. Yalnızca sana "
    "verilen kaynak metinlerdeki bilgilerden soru üret. Yanıtın SADECE geçerli "
    "JSON olsun: açıklama, giriş cümlesi, kod bloğu işareti veya kaynak atfı ekleme. "
    "Her öğe question, options ve answer alanlarına sahip olsun; answer mutlaka "
    "kısa bir string olsun, nesne veya ek alan kullanma. JSON'u tamamlamadan kesme."
)

SYSTEM_PROMPT_QUIZ_EN = (
    "You generate study questions from course notes. Base every question only on "
    "the provided source excerpts. Your reply must be ONLY valid JSON: no "
    "explanation, no preamble, no code fences, no source citations. Every item "
    "must have question, options, and answer; answer must be a short string, "
    "never an object or extra field. Do not stop before closing the JSON."
)

# Yalnızca thinking modelleri için (bkz. config.py:model_has_thinking) sistem
# prompt'una eklenen ek bastırma talimatı. Düşünmeyen bir modelde bu cümle
# gereksiz -- ve modelin var olmayan bir davranışa odaklanmasına yol açabilir --
# bu yüzden koşullu eklendi.
_NO_THINK_INSTRUCTION_TR = (
    " ÖNEMLİ: <think> etiketi veya herhangi bir 'akıl yürütme' bloğu KULLANMA; "
    "uzun uzadıya düşünmeden doğrudan nihai cevabı yaz."
)
_NO_THINK_INSTRUCTION_EN = (
    " IMPORTANT: Do NOT use <think> tags or any reasoning block; write the final "
    "answer directly without lengthy internal deliberation."
)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """Qwen3'ün ürettiği `<think>...</think>` akıl yürütme bloğunu temizler.

    `/no_think` soft-switch'i genelde bu bloğun hiç üretilmesini engelliyor,
    ama garanti değil (özellikle daha karmaşık sorularda model yine de
    düşünebiliyor). Bu fonksiyon, ne olursa olsun kullanıcıya/JSON parser'a
    temiz metin gitmesini sağlayan bir güvenlik ağı.
    """
    return _THINK_BLOCK_RE.sub("", text).strip()


def build_prompt(
    question: str,
    context_chunks: list[str],
    language: str,
    has_context: bool | None = None,
    system_prompt: str | None = None,
) -> list[dict]:
    """Sohbet mesajlarını oluşturur.

    `has_context=False` verilirse (ya da `context_chunks` boşsa) "kaynak
    bulunamadı" moduna geçilir: context hiç gönderilmez ve model, bilginin
    dosyalardan gelmediğini belirterek genel bilgiyle cevap verir (bkz.
    `SYSTEM_PROMPT_NO_CONTEXT_TR`). Böylece kullanıcı, dosyalarında olmayan
    bir şey sorduğunda boş/kaçamak bir cevap yerine işe yarar bir cevap alıyor.

    `system_prompt` ile varsayılan sistem prompt'u değiştirilebilir (bkz.
    services/quiz.py: quiz çıktısı JSON olarak parse edildiği için oradaki
    "kaynak atfı yap" talimatı zararlı).
    """
    is_tr = language == "tr"
    if has_context is None:
        has_context = bool(context_chunks)

    if system_prompt is not None:
        system = system_prompt
    elif has_context and context_chunks:
        system = SYSTEM_PROMPT_TR if is_tr else SYSTEM_PROMPT_EN
    else:
        system = SYSTEM_PROMPT_NO_CONTEXT_TR if is_tr else SYSTEM_PROMPT_NO_CONTEXT_EN

    # Akıl yürütme bastırma talimatı ve `/no_think` soft-switch'i YALNIZCA
    # thinking modelinde eklenir (bkz. config.py:model_has_thinking).
    if settings.model_has_thinking:
        system += _NO_THINK_INSTRUCTION_TR if is_tr else _NO_THINK_INSTRUCTION_EN

    question_label = "Soru" if is_tr else "Question"
    if has_context and context_chunks:
        context = "\n\n---\n\n".join(
            f"[Kaynak {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
        )
        sources_label = "Kaynaklar" if is_tr else "Sources"
        user_content = f"{sources_label}:\n{context}\n\n{question_label}: {question}"
    else:
        user_content = f"{question_label}: {question}"

    if settings.model_has_thinking:
        # Qwen3 hibrit bir "thinking" modeli; kullanıcı mesajının sonuna
        # eklenen /no_think soft-switch'i chat template seviyesinde akıl
        # yürütme bloğunu büyük ölçüde bastırıyor (bkz. QwenLM/Qwen3
        # discussion #1300). Düşünmeyen modellerde bu metin cevaba
        # sızabildiği için eklenmiyor.
        user_content += " /no_think"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


# Foundry Local'de gözlemlenen bir başka arıza modu: model (özellikle
# GPU'daki quantize sürümlerde, repetition penalty olmadan) bazen bozuk bir
# tekrar döngüsüne girip aynı kısa kelimeyi/harfi onlarca-yüzlerce kez art
# arda üretiyor (gözlemlenen: "Otonom Otonom Otonom ... O O O O ..."). Bu iki
# şekilde ele alınıyor: (1) `frequency_penalty`/`presence_penalty` ile modele
# tekrarı caydırıcı bir sinyal veriliyor (standart OpenAI parametreleri;
# Foundry Local'in backend'i desteklemiyorsa muhtemelen sessizce yok
# sayılır, zararı olmaz); (2) buna rağmen olursa `_RepetitionGuard` akışı
# erken kesip anlaşılır bir hata veriyor — kullanıcıya yüzlerce token'lık
# anlamsız tekrar göstermek yerine.
_REPETITION_FREQUENCY_PENALTY = 0.4
_REPETITION_PRESENCE_PENALTY = 0.2


def _is_blank(text: str) -> bool:
    """Boş ya da yalnızca boşluk karakterlerinden oluşan metni tespit eder.

    `generate_answer`/`generate_answer_stream`'in, modelin görünür hiçbir
    metin üretmediği durumu (bkz. `EmptyAnswer`) sessizce boş bir cevap
    olarak geçirmek yerine net bir hataya çevirmesi için kullanılıyor."""
    return not text.strip()


# Gözlemlenen bir sonraki arıza modu (`_is_blank`'ın YAKALAYAMADIĞI):
# model `<think>` bloğunu düzgün kapatıp görünür bir cevaba geçiyor, ama bu
# "cevap" tek bir noktalama işareti (`"."`) gibi anlamsız derecede kısa
# kalıyor -- muhtemelen 1500 token bütçesinin neredeyse tamamı akıl
# yürütmede tükendiği, gerçek cevaba yalnızca bir-iki karakterlik yer
# kaldığı için. Kullanıcı için pratik sonuç `_is_blank` ile aynı: gösterecek
# bir cevap yok. Bu yüzden ikisini de aynı `EmptyAnswer`'a çeviriyoruz (bkz.
# `_looks_too_short`, `_stream_with_warmup`).
_MIN_MEANINGFUL_ANSWER_CHARS = 4


def _looks_too_short(text: str) -> bool:
    """`_is_blank`'ın ıskaladığı, gözlemlenen "anlamsız derecede kısa cevap"
    arıza modunu (bkz. yukarıdaki not) tespit eder."""
    return 0 < len(text.strip()) < _MIN_MEANINGFUL_ANSWER_CHARS


# ARAŞTIRILDI (bkz. microsoft/foundry-local GitHub issue #808, "Add ability
# to modify chat template kwargs"): Foundry Local'in REST API'si şu anda
# `chat_template_kwargs`/`extra_body` alanını HİÇ TANIMIYOR -- resmi REST
# referansındaki kabul edilen alan listesinde (model, messages, temperature,
# top_p, max_tokens, stop, tools, ep, ttl, random_seed, ...) bu alan yok.
# Yani bu, `enable_thinking=False`'ın Foundry Local'in altındaki ONNX
# Runtime GenAI motoruna hiç ulaşmadığı, sessizce yok sayıldığı anlamına
# geliyor -- `BadRequestError` de FIRLATMIYOR (aksi halde fallback zaten
# devreye girerdi), bu yüzden `/no_think` metin ipucundan başka gerçek bir
# "akıl yürütmeyi kapat" yolumuz şu an YOK. (Qwen3'ün vLLM/SGLang'deki
# resmi alternatifi -- boş bir `<think>\n\n</think>\n\n` ile asistan
# turunu "prefill" edip `continue_final_message` kullanmak -- da Foundry
# Local'in mesaj şemasında karşılığı olmadığı için burada uygulanamıyor.)
# Yine de bu çağrıyı KALDIRMIYORUZ: risksiz (hata fırlatmıyor, zararı yok)
# ve #808 issue'su ileride çözülürse otomatik olarak işe yarayacak.
def _foundry_extra_body() -> dict:
    """OpenAI şemasında OLMAYAN, Foundry Local'e özgü alanlar.

    OpenAI Python SDK'sinin `create()` imzası tiplenmiş; burada olmayan bir
    anahtar kelime argümanı sunucuya hiç gitmeden `TypeError` ile
    reddediliyor. Foundry Local'in kabul ettiği ama OpenAI'da bulunmayan
    alanlar (`ep`, `ttl`, `top_k`, `random_seed`) bu yüzden `extra_body`
    içinde gönderilmek zorunda.
    """
    extra: dict = {}
    if settings.foundry_execution_provider:
        # bkz. core/config.py:foundry_execution_provider -- 8 GB kartta
        # KV cache OOM'unu aşmanın çalışan tek yolu.
        extra["ep"] = settings.foundry_execution_provider
    return extra


def _create_chat_completion(client, **kwargs):
    """`client.chat.completions.create(**kwargs)`'ı, mümkünse önce Qwen3'ün
    resmi `enable_thinking=False` anahtarıyla dener; desteklenmiyorsa
    (`BadRequestError`) aynı isteği bu ekstra alan olmadan tekrar dener.

    `client`'ı parametre olarak alıyoruz ki gerçek bir Foundry Local
    bağlantısına ihtiyaç duymadan, sahte (fake) bir client ile bu geri
    dönme (fallback) davranışını birim testiyle doğrulayabilelim (bkz.
    tests/test_llm_utils.py).
    """
    from openai import BadRequestError

    extra = _foundry_extra_body()

    # Düşünmeyen (instruct) bir model kullanılıyorsa bastırılacak bir akıl
    # yürütme zaten yok: `enable_thinking` denemesini hiç yapmıyoruz.
    if not settings.model_has_thinking:
        if extra:
            kwargs["extra_body"] = extra
        return client.chat.completions.create(**kwargs)

    try:
        return client.chat.completions.create(
            extra_body={**extra, "chat_template_kwargs": {"enable_thinking": False}},
            **kwargs,
        )
    except BadRequestError:
        if extra:
            kwargs["extra_body"] = extra
        return client.chat.completions.create(**kwargs)


# `enable_thinking`'i Foundry Local'e gerçekten ulaştıramadığımız
# doğrulandığı için (bkz. yukarıdaki not), akıl yürütmeyi PROAKTİF olarak
# engelleyemiyoruz -- elimizdeki tek gerçek kaldıraç, akıl yürütme +
# gerçek cevaba yetecek kadar `max_tokens` payı bırakmak. Bu yüzden sabit
# tek bir bütçe yerine, İLK denemede hızlı/ucuz bir bütçeyle başlayıp
# (çoğu soru kısa akıl yürütmeyle bitiyor), başarısız olursa (bkz.
# EmptyAnswer/_looks_too_short) çok daha geniş bir bütçeyle YENİDEN
# deniyoruz -- ilk denemedeki AYNI (küçük) bütçeyle tekrar denemenin hiçbir
# faydası olmadığı gözlemlendi (Foundry Local'in akıl yürütmeyi bastırma
# yeteneği yok, bu yüzden ikinci deneme de aynı şekilde tıkanabiliyor).
# İkinci denemedeki 4096, Foundry Local'in `max_tokens` verilmediğinde
# kullandığı ~2048'lik varsayılanın bile üzerinde -- gözlemlenen bazı
# vakalarda 1500'ün de yetersiz kaldığı görüldüğü için kasıtlı olarak
# cömert tutuldu (bunun bedeli yalnızca -- yalnızca ilk deneme
# başarısız olduğunda -- ekstra üretim süresi).
# Bütçeler artık config'den geliyor (bkz. config.py:answer_token_budgets):
# düşünmeyen modelde TEK ve küçük bir bütçe (varsayılan 800) yeterli --
# üretilen her token doğrudan görünür cevap olduğu için "bütçeyi akıl
# yürütmede tüketme" arızası hiç oluşmuyor ve üretim çok daha hızlı bitiyor.
# Thinking modelinde eski iki kademeli (1500 -> 4096) eskalasyon korunuyor.
def _answer_token_budgets() -> list[int]:
    return settings.answer_token_budgets

# Hem boş (bkz. EmptyAnswer) hem de "anlamsız derecede kısa" (bkz.
# _looks_too_short) cevaplar, gözlemlenen davranışa göre olasılıksal bir
# arıza -- aynı soruyu tekrar denemek (temperature=0.2 olsa da örnekleme
# hâlâ deterministik değil), ÖZELLİKLE de yukarıdaki gibi çok daha geniş
# bir bütçeyle, çoğu zaman gerçek bir cevap üretiyor. Kullanıcının elle
# "gönder"e tekrar tekrar basmasını (gözlemlenen bir kullanıcı ekran
# görüntüsünde tam olarak bu olmuştu: aynı soru art arda 3 kez
# gönderilmişti) gereksiz kılmak için `_answer_token_budgets`'teki her
# bütçeyi sırayla otomatik deniyoruz. Streaming tarafında bu YALNIZCA
# güvenli: bkz.
# `_stream_with_warmup` -- eşiğe ulaşılamadıysa çağırana HİÇBİR token henüz
# yayınlanmamış oluyor, bu yüzden yeni bir deneme başlatmak kullanıcıya
# hiçbir zaman yarım/anlamsız bir metin göstermiyor.


def generate_answer(
    question: str,
    context_chunks: list[str],
    language: str = "tr",
    has_context: bool | None = None,
) -> str:
    """Foundry Local üzerinden yerel modeli çalıştırıp cevap üretir."""
    manager = _get_manager()
    model_id = _get_model_id()
    messages = build_prompt(question, context_chunks, language, has_context)
    budgets = _answer_token_budgets()

    # foundry-local-sdk, OpenAI Python istemcisiyle uyumlu bir endpoint döner.
    from openai import OpenAI

    client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key)

    last_error: EmptyAnswer | None = None
    for attempt, max_tokens in enumerate(budgets, start=1):
        response = _create_chat_completion(
            client,
            model=model_id,
            messages=messages,
            temperature=0.2,
            frequency_penalty=_REPETITION_FREQUENCY_PENALTY,
            presence_penalty=_REPETITION_PRESENCE_PENALTY,
            # bkz. _answer_token_budgets ve _create_chat_completion'daki notlar.
            max_tokens=max_tokens,
        )
        answer = strip_think(response.choices[0].message.content)
        if _is_blank(answer) or _looks_too_short(answer):
            usage = getattr(response, "usage", None)
            logger.warning(
                "Boş/anlamsız-kısa cevap (deneme %d/%d, max_tokens=%d, "
                "completion_tokens=%s, finish_reason=%s)",
                attempt,
                len(budgets),
                max_tokens,
                getattr(usage, "completion_tokens", "?"),
                getattr(response.choices[0], "finish_reason", "?"),
            )
            last_error = EmptyAnswer(
                "Model bu soru için görünür bir cevap üretmedi (muhtemelen "
                "`/no_think`'e rağmen tüm token bütçesini akıl yürütmede tükettim, "
                "ya da geriye anlamsız derecede kısa bir metin kaldı). Otomatik "
                "yeniden deneme(ler) de sonuç vermedi -- soruyu tekrar dene ya da "
                "biraz daha basit ifade et."
            )
            continue
        if _looks_degenerate(answer):
            raise DegenerateOutput(
                "Model bozuk bir tekrar döngüsüne girdi (aynı kelime ya da aynı kelime "
                "öbeği kendini tekrarlayıp durdu). Soruyu farklı ifade edip tekrar dene."
            )
        return answer

    assert last_error is not None
    raise last_error


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _partial_open_tag_len(text: str) -> int:
    """`text`'in sonundaki, `<think>` etiketinin yarım kalmış bir öneki
    olabilecek en uzun parçanın uzunluğunu döner (yoksa 0).

    Streaming'de model çıktısı küçük parçalar (delta) halinde geliyor; bir
    parça tam olarak "<thi" ile bitip bir sonraki parça "nk>..." ile devam
    edebilir. Bu kontrol olmadan "<thi" hemen görünür metin olarak
    kullanıcıya gösterilir, ardından "nk>" gelince etiket asla tanınmaz.
    Bu yüzden her adımda, tamponun sonunda `<think>`'in yarım kalmış bir
    öneki varsa onu bir sonraki parçayı bekleyerek tamponda tutuyoruz.
    """
    limit = min(len(_THINK_OPEN) - 1, len(text))
    for length in range(limit, 0, -1):
        if _THINK_OPEN.startswith(text[-length:]):
            return length
    return 0


class _ThinkStreamStripper:
    """`strip_think`'in streaming (parça parça gelen metin) karşılığı.

    Qwen3'ün `<think>...</think>` bloğu (bkz. `strip_think`) streaming
    modunda token token geliyor; etiketin kendisi bile birden fazla delta'ya
    bölünmüş olabilir. Bu sınıf gelen her parçayı (`feed`) bir tamponda
    biriktirip yalnızca `<think>` bloklarının DIŞINDA kalan, kesin olarak
    tamamlanmış metni döner; belirsiz (yarım etiket olabilecek) kısmı bir
    sonraki parçaya kadar tamponda tutar. `flush`, akış bittiğinde tamponda
    kalan her şeyi güvenli şekilde sonuçlandırır.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False

    def feed(self, delta: str) -> str:
        self._buffer += delta
        out: list[str] = []
        while True:
            if self._in_think:
                idx = self._buffer.find(_THINK_CLOSE)
                if idx == -1:
                    return "".join(out)
                self._buffer = self._buffer[idx + len(_THINK_CLOSE):]
                self._in_think = False
                continue

            idx = self._buffer.find(_THINK_OPEN)
            if idx == -1:
                hold = _partial_open_tag_len(self._buffer)
                emit_len = len(self._buffer) - hold
                out.append(self._buffer[:emit_len])
                self._buffer = self._buffer[emit_len:]
                return "".join(out)

            out.append(self._buffer[:idx])
            self._buffer = self._buffer[idx + len(_THINK_OPEN):]
            self._in_think = True

    def flush(self) -> str:
        """Akış bittiğinde çağrılır. Yarım kalmış `<think>` öneki (asla tam
        etikete dönüşmedi) varsa düz metin olarak döner; ama tamamen açılmış
        ve hiç kapanmamış bir `<think>` bloğunun içeriği kasıtlı olarak
        SİLİNİR (kullanıcıya yarım/kesik akıl yürütme metni göstermemek
        için) — `strip_think`'in tamamlanmış girdide etiket arasını hep
        gizlemesiyle aynı davranış.
        """
        remaining = self._buffer
        self._buffer = ""
        return "" if self._in_think else remaining


# Tekrar döngüsünün İKİNCİ (ve pratikte daha sık görülen) biçimi: model tek
# bir kelimeyi değil, çok kelimeli bir ÖBEĞİ döngüye alıyor. Gözlemlenen
# gerçek çıktı (kullanıcı ekran görüntüsü):
#
#   "... çapkalarda kullanma ve kullanma için kullanma, çapkalarda kullanma
#    ve kullanma için kullanma, çapkalarda kullanma ve kullanma için ..."
#
# Burada AYNI kelime hiçbir zaman art arda gelmiyor -- altı kelimelik bir
# öbek dönüp duruyor. Bu yüzden yalnızca ardışık aynı kelimeyi sayan eski
# koruma hiç tetiklenmiyor ve kullanıcı ekranı dolduran anlamsız bir metin
# görüyordu. Aşağıdaki periyot taraması bunu yakalıyor: son kelimelerin
# kuyruğu, uzunluğu `period` olan bir bloğun `min_phrase_repeats` kez
# ARDIŞIK tekrarından ibaretse döngü var demektir.
#
# Eşikler bilinçli olarak muhafazakâr: iki veya daha çok kelimelik bir
# öbeğin arada hiçbir şey olmadan dört kez üst üste tekrarlanması normal
# metinde (Türkçe ya da İngilizce) pratikte görülmez.
_MAX_PHRASE_PERIOD = 12
_MIN_PHRASE_REPEATS = 4

# Kelimeleri karşılaştırırken noktalama ve büyük/küçük harf farkını yok
# sayıyoruz: gözlemlenen döngüde aynı öbek bazen "kullanma," bazen
# "kullanma" olarak geliyor ve ham karşılaştırma periyodu kaçırıyor.
_WORD_NORMALIZE_RE = re.compile(r"[^\w]+", re.UNICODE)


def _normalize_word(word: str) -> str:
    return _WORD_NORMALIZE_RE.sub("", word.lower()) or word.lower()


class _RepetitionGuard:
    """Modelin bozuk bir tekrar döngüsüne girip girmediğini izler (bkz.
    `_REPETITION_FREQUENCY_PENALTY`'nin üzerindeki not).

    İki ayrı döngü biçimini yakalar:

    1. **Tek kelime döngüsü.** Aynı kelime `max_consecutive_repeats` kez üst
       üste gelirse (gözlemlenen: "Otonom Otonom Otonom ...").
    2. **Öbek döngüsü.** Uzunluğu `2..max_phrase_period` arasında olan bir
       kelime öbeği `min_phrase_repeats` kez ardışık tekrarlanırsa (bkz.
       yukarıdaki not) -- tek kelime kuralının GÖREMEDİĞİ biçim.

    Metin küçük parçalar (delta) halinde gelebileceği için, henüz
    tamamlanmamış (boşlukla bitmeyen) son kelimeyi bir sonraki `feed`
    çağrısına kadar tamponda tutar.
    """

    def __init__(
        self,
        max_consecutive_repeats: int = 10,
        max_phrase_period: int = _MAX_PHRASE_PERIOD,
        min_phrase_repeats: int = _MIN_PHRASE_REPEATS,
    ) -> None:
        self._max_repeats = max_consecutive_repeats
        self._max_phrase_period = max_phrase_period
        self._min_phrase_repeats = min_phrase_repeats
        self._buffer = ""
        self._last_word: str | None = None
        self._run_length = 0
        # Periyot taraması için gereken en uzun kuyruk; bundan eskisini
        # tutmanın faydası yok (akış boyunca bellek sabit kalıyor).
        self._window = max_phrase_period * min_phrase_repeats
        self._recent: list[str] = []

    def feed(self, text: str) -> bool:
        self._buffer += text
        *complete_words, self._buffer = self._buffer.split(" ")
        for word in complete_words:
            word = word.strip()
            if not word:
                continue

            if word == self._last_word:
                self._run_length += 1
            else:
                self._last_word = word
                self._run_length = 1
            if self._run_length >= self._max_repeats:
                return True

            self._recent.append(_normalize_word(word))
            if len(self._recent) > self._window:
                del self._recent[: len(self._recent) - self._window]
            if self._has_phrase_loop():
                return True
        return False

    def _has_phrase_loop(self) -> bool:
        """Kuyruk, tek bir bloğun ardışık tekrarından mı ibaret?"""
        n = len(self._recent)
        for period in range(2, self._max_phrase_period + 1):
            span = period * self._min_phrase_repeats
            if n < span:
                # `span` periyotla birlikte büyüdüğü için daha uzun
                # periyotlara bakmanın anlamı yok.
                break
            tail = self._recent[n - span :]
            block = tail[:period]
            # Tek kelimenin tekrarı (ör. "Otonom Otonom Otonom ...") her
            # periyot için "öbek döngüsü" gibi görünür, ama o durumun kendi
            # (daha yüksek ve bilinçli seçilmiş) eşiği var:
            # `max_consecutive_repeats`. Burada yakalarsak o eşiği sessizce
            # 2 * min_phrase_repeats'e indirmiş oluruz -- bu kural yalnızca
            # ÇOK KELİMELİ öbekler için.
            if len(set(block)) == 1:
                continue
            if all(
                tail[i * period : (i + 1) * period] == block
                for i in range(1, self._min_phrase_repeats)
            ):
                return True
        return False


def _looks_degenerate(text: str) -> bool:
    """Tamamlanmış (streaming olmayan) bir cevabın bozuk bir tekrar
    döngüsü içerip içermediğini kontrol eder — `_RepetitionGuard`'ı tüm
    metni tek seferde besleyerek kullanır (bkz. `generate_answer`)."""
    guard = _RepetitionGuard()
    return guard.feed(text + " ")


def _stream_and_strip(events):
    """OpenAI streaming chunk'larından (`.choices[0].delta.content` alanı
    olan nesnelerden) `<think>` bloğu temizlenmiş metin parçaları üretir.

    Gözlemlenen bir Foundry Local davranışı: üretim bittikten hemen sonra
    (veya bazen sırasında) bağlantı, chunked-encoding'in kapanış işaretini
    göndermeden kesiliyor — bu httpx tarafında `RemoteProtocolError: peer
    closed connection without sending complete message body (incomplete
    chunked read)` olarak görünüyor (Foundry Local hâlâ olgunlaşmamış bir
    servis; dosyanın başındaki diğer CLI/servis notlarına bakın). Bu durumda
    zaten akmış olan metni kaybetmek yerine akışı sessizce (hata fırlatmadan)
    sonlandırıyoruz — kullanıcı büyük ihtimalle cevabın tamamını ya da
    neredeyse tamamını almış oluyor. Hiç içerik gelmeden bağlantı koparsa bu
    gerçek bir sorunun işareti, o durumda hatayı olduğu gibi yükseltiyoruz
    (çağıran `generate_answer_stream` bunu kullanıcıya gösterilebilir bir
    `FoundryNotAvailable`'a çeviriyor).

    Bu fonksiyon, döngü + hata yönetimini `generate_answer_stream`'den
    kasıtlı olarak ayırıyor: böylece gerçek bir Foundry Local/HTTP bağlantısı
    kurmadan, sahte (fake) bir `events` iterable'ıyla saf birim testi
    yazılabiliyor (bkz. tests/test_llm_utils.py).
    """
    stripper = _ThinkStreamStripper()
    received_any = False
    try:
        for event in events:
            # Gözlemlenen bir başka Foundry Local tuhaflığı: OpenAI'ın gerçek
            # API'sinde `choices` yalnızca `stream_options={"include_usage":
            # True}` istendiğinde son bir "usage" chunk'ında boş liste olarak
            # gelir (biz bunu istemiyoruz), ama Foundry Local akışın bir
            # yerinde (gözlemlenen: üretimin hemen ardından) yine de boş
            # `choices` listeli bir chunk gönderebiliyor. Bunu `[0]` ile
            # indekslemek `IndexError` ile çöküyordu; içerik taşımayan bu
            # tür chunk'ları sessizce atlıyoruz.
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if not delta:
                continue
            received_any = True
            piece = stripper.feed(delta)
            if piece:
                yield piece
    except TRANSPORT_ERRORS:
        if not received_any:
            raise

    tail = stripper.flush()
    if tail:
        yield tail


def _stream_with_warmup(pieces, min_chars: int = _MIN_MEANINGFUL_ANSWER_CHARS):
    """`_stream_and_strip`'in ürettiği (zaten `<think>` temizlenmiş) parça
    dizisinin önüne bir "ısınma" (warm-up) tamponu ekler: çağırana hiçbir
    şey YAYINLAMADAN önce en az `min_chars` kadar görünür (boşluk
    olmayan) karakter birikmesini bekler.

    Amaç, `_is_blank`'ın kaçırdığı gözlemlenen bir arıza modu: model
    `<think>` bloğunu kapatıp görünür bir cevaba geçiyor ama bu "cevap" tek
    bir noktalama işareti (`"."`) gibi anlamsız derecede kısa kalıyor.
    Isınma eşiğine hiç ulaşılmazsa (akış anlamsız kısalıkta biterse) bu
    fonksiyon HİÇBİR ŞEY yield etmez -- bu da `generate_answer_stream`'in,
    kullanıcıya o ana kadar hiçbir token göndermeden (çünkü hiçbiri
    yayınlanmadı) sessizce yeni bir deneme başlatabilmesini sağlıyor.

    Eşik aşıldıktan sonra normal, parça parça (canlı) akış devam eder --
    yalnızca ilk görünür parça, birkaç delta'nın birleşimi olarak biraz
    daha "toplu" gelir; bu, sıradan (yeterince uzun) cevaplarda fark
    edilmeyecek kadar küçük bir gecikme.
    """
    buffer = ""
    warmed_up = False
    for piece in pieces:
        if warmed_up:
            yield piece
            continue
        buffer += piece
        if len(buffer.strip()) < min_chars:
            continue
        warmed_up = True
        yield buffer
        buffer = ""
    # NOT: `buffer`'da eşiğe hiç ulaşmamış bir kalıntı varsa (akış anlamsız
    # kısalıkta bittiyse) KASITLI OLARAK yayınlanmıyor.


def generate_answer_stream(
    question: str,
    context_chunks: list[str],
    language: str = "tr",
    has_context: bool | None = None,
):
    """`generate_answer`'ın streaming (token token) karşılığı.

    Foundry Local'in OpenAI-uyumlu endpoint'i `stream=True` ile parça parça
    (delta) bir akış döner; bunu burada bir Python generator'ına çevirip
    `<think>` bloğunu akış sırasında (`_ThinkStreamStripper`) temizleyerek
    yalnızca kullanıcıya gösterilecek metin parçalarını yield ediyoruz.
    Bağlantı üretim sırasında/sonunda beklenmedik şekilde koparsa (bkz.
    `_stream_and_strip`) ve hiç içerik gelmediyse `FoundryNotAvailable`
    fırlatır. Model bozuk bir tekrar döngüsüne girerse (bkz.
    `_RepetitionGuard`) o ana kadar akan (muhtemelen sağlam) metni yayınlayıp
    akışı `DegenerateOutput` ile keser. Akış hatasız bitmesine rağmen hiç
    görünür parça üretilmediyse (bkz. `EmptyAnswer`) bunu da açık bir hataya
    çevirir. `/ask/stream` üçünü de kullanıcıya gösterilebilir bir hata
    olayına çevirir.
    """
    manager = _get_manager()
    model_id = _get_model_id()
    messages = build_prompt(question, context_chunks, language, has_context)
    budgets = _answer_token_budgets()

    from openai import OpenAI

    client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key)

    # bkz. _answer_token_budgets'in üzerindeki not: bu döngü, listedeki her
    # bütçeyi (küçükten büyüğe) sırayla dener, ve bir sonraki denemeye
    # YALNIZCA hiçbir token henüz çağırana yayınlanmamışsa geçer (bkz.
    # _stream_with_warmup) -- yani kullanıcı hiçbir zaman yarım/anlamsız bir
    # cevap görmeden, sanki tek bir istekmiş gibi (sadece biraz daha uzun
    # süren) bir sonuç alır.
    last_error: EmptyAnswer | None = None
    for attempt, max_tokens in enumerate(budgets, start=1):
        stream = _create_chat_completion(
            client,
            model=model_id,
            messages=messages,
            temperature=0.2,
            frequency_penalty=_REPETITION_FREQUENCY_PENALTY,
            presence_penalty=_REPETITION_PRESENCE_PENALTY,
            # bkz. _answer_token_budgets ve _create_chat_completion'daki notlar.
            max_tokens=max_tokens,
            stream=True,
        )

        guard = _RepetitionGuard()
        produced_output = False
        try:
            for piece in _stream_with_warmup(_stream_and_strip(stream)):
                produced_output = True
                if guard.feed(piece):
                    raise DegenerateOutput(
                        "Model bozuk bir tekrar döngüsüne girdi (aynı kelime ya da "
                        "aynı kelime öbeği kendini tekrarlayıp durdu); cevap erken "
                        "kesildi. Soruyu farklı ifade edip tekrar dene."
                    )
                yield piece
        except TRANSPORT_ERRORS as exc:
            # Buraya YALNIZCA hiç içerik gelmeden bağlantı koptuysa
            # düşülüyor (bkz. `_stream_and_strip`: içerik akmışsa hata
            # yutuluyor). Bu ayrımın teşhis değeri yüksek: Foundry Local
            # 200 OK + `text/event-stream` başlığını gönderdikten SONRA
            # ölmüş demektir, yani istek geçerliydi.
            #
            # ÖLÇÜLDÜ: bu durumun sebebi VRAM. Foundry Local'in kendi
            # log'unda (`foundry server logs`) karşılığı şu:
            #
            #   OnnxRuntimeGenAIException: CUDA error in CudaMallocArray
            #     ... - out of memory
            #     at Microsoft.ML.OnnxRuntimeGenAI.Generator..ctor(...)
            #
            # Çökme `Generator` KURULURKEN oluyor, yani modelin ağırlıkları
            # zaten yüklüyken isteğin KV cache'i için yer kalmadığında. Model
            # "başarıyla yüklendi" dediği için sorun dışarıdan görünmüyor.
            # Kullanıcıya "bağlantı kesildi" demek doğru ama işe yaramaz;
            # asıl sebebi ve ne yapabileceğini söylüyoruz.
            raise FoundryNotAvailable(
                "Foundry Local üretime hiç başlayamadan bağlantıyı kesti: GPU belleği "
                "(VRAM) bu isteğin KV cache'i için yetmedi. Doğrulamak için "
                "`foundry server logs` çıktısında \"CudaMallocArray - out of memory\" "
                "satırını ara.\n\n"
                "ÖLÇÜLDÜ: ANSWER_MAX_TOKENS / MAX_CONTEXT_* değerlerini düşürmek bunu "
                "ÇÖZMÜYOR — max_tokens 800'den 500'e ve bağlam 6552'den 3430 bayta "
                "indirildiğinde hata birebir aynı kaldı. Tahsis, isteğin uzunluğuna "
                "değil modelin tam bağlam penceresine göre yapılıyor.\n\n"
                "Çalışan iki çözüm: (1) `.env`'de FOUNDRY_EXECUTION_PROVIDER=cpu — "
                "yavaş ama OOM olmuyor; (2) daha küçük bir model (`foundry model list` "
                "ile bak, `.env`'de FOUNDRY_MODEL_ALIAS ile değiştir)."
            ) from exc

        if produced_output:
            return

        # `_stream_with_warmup` hiçbir şey yayınlamadı: model ya tamamen
        # boş kaldı ya da geriye anlamsız derecede kısa (ör. ".") bir metin
        # bıraktı. Henüz çağırana HİÇBİR token gitmediği için bir sonraki
        # (daha geniş bütçeli) denemeye -- varsa -- sessizce geçmek güvenli.
        logger.warning(
            "Boş/anlamsız-kısa akış (deneme %d/%d, max_tokens=%d)",
            attempt,
            len(budgets),
            max_tokens,
        )
        last_error = EmptyAnswer(
            "Model bu soru için görünür bir cevap üretmedi (muhtemelen `/no_think`'e "
            "rağmen tüm token bütçesini akıl yürütmede tükettim, `<think>` bloğu hiç "
            "kapanmadan token sınırına ulaşıldı, ya da geriye anlamsız derecede kısa "
            "bir metin kaldı). Otomatik yeniden deneme(ler) de sonuç vermedi -- soruyu "
            "tekrar dene ya da biraz daha basit ifade et."
        )

    assert last_error is not None
    raise last_error


def warmup() -> None:
    """Foundry Local servisini ve modeli, ilk kullanıcı sorusundan ÖNCE
    hazır hâle getirir.

    Eskiden `_get_manager`/`_get_model_id` (ve dolayısıyla `foundry server
    start` + `foundry model load`) ilk `/ask` isteğinin İÇİNDE çalışıyordu:
    kullanıcı ilk sorusunu sorduğunda, modelin diskten belleğe yüklenmesi
    dahil onlarca saniye bekliyordu ve bu "uygulama çok yavaş" izlenimi
    veriyordu. Bu fonksiyon backend açılırken arka planda bir kez çağrılır
    (bkz. main.py); ikisi de `lru_cache`'li olduğu için sonraki çağrılar
    bedava.

    Hata yutuluyor: Foundry Local kapalıysa bile backend ayağa kalkmalı --
    kullanıcı ilk soruyu sorduğunda zaten anlaşılır bir `FoundryNotAvailable`
    hatası alacak.
    """
    try:
        _get_manager()
        _get_model_id()
        logger.info("Foundry Local hazır (model: %s)", settings.foundry_model_alias)
    except Exception as exc:  # pragma: no cover - ortama bağlı
        logger.warning("Model ön ısıtma başarısız (ilk soruda tekrar denenecek): %s", exc)
