"""Doküman özeti üretimi.

Bir dokümanın TAMAMINI özetlemek, `/ask`'ten farklı bir problem: orada
sorguya en alakalı birkaç parça seçiliyor, burada ise hiçbir şeyi atlamamak
gerekiyor. Uzun bir ders notu yüzlerce parçaya bölünebildiği için hepsini tek
istekte göndermek modelin context penceresini aşar (ve aşmasa bile üretimi
dakikalarca sürdürür).

Bu yüzden klasik **map-reduce** yaklaşımı kullanılıyor:

    map    : doküman ardışık gruplara bölünür, her grup ayrı ayrı özetlenir
    reduce : ara özetler birleştirilip TEK bir nihai özete indirgenir

Doküman zaten tek bir gruba sığıyorsa (kısa dosyalar — pratikte çoğu) map
adımı atlanır ve doğrudan tek bir çağrı yapılır; gereksiz bir ikinci tur
maliyeti ödenmez.
"""
from __future__ import annotations

import logging

from backend.app.core.config import settings
from backend.app.services.llm import (
    is_capacity_error,
    raise_if_gpu_context_lost,
    _create_chat_completion,
    _get_manager,
    _get_model_id,
    _stream_and_strip,
    _stream_with_warmup,
    strip_think,
)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_SUMMARY_TR = (
    "Ders notlarını öğrenciler için özetleyen bir asistansın. Yalnızca sana "
    "verilen metindeki bilgileri kullan; dışarıdan bilgi ekleme, yorum katma. "
    "Özet, notu okumamış birinin konuyu anlayabileceği kadar kapsamlı ama "
    "gereksiz tekrardan arınmış olsun. Tanımları, formülleri ve önemli "
    "ayrımları mutlaka koru. Markdown kullan: ana konuları '## Başlık' ile "
    "böl, maddelenebilir bilgileri liste yap, anahtar terimleri **kalın** yaz."
)

SYSTEM_PROMPT_SUMMARY_EN = (
    "You summarize course notes for students. Use only the information in the "
    "text you are given; do not add outside knowledge or commentary. The "
    "summary should be complete enough that someone who has not read the notes "
    "can follow the topic, but free of redundancy. Always keep definitions, "
    "formulas and important distinctions. Use markdown: split main topics with "
    "'## Heading', use lists for enumerable facts, **bold** for key terms."
)

# Ara (map adımı) özetler için: bunlar kullanıcıya gösterilmiyor, yalnızca
# reduce adımına girdi olacak. Bu yüzden biçimlendirme değil, BİLGİ YOĞUNLUĞU
# isteniyor — markdown başlıkları burada sadece yer kaplardı.
SYSTEM_PROMPT_PARTIAL_TR = (
    "Sana bir ders notunun bir BÖLÜMÜ veriliyor. Bu bölümdeki bilgileri, hiçbir "
    "önemli tanımı/formülü/ayrımı atlamadan, sıkı ve maddeler hâlinde çıkar. "
    "Giriş cümlesi yazma, yorum ekleme, metinde olmayan bilgi uydurma."
)

SYSTEM_PROMPT_PARTIAL_EN = (
    "You are given a SECTION of course notes. Extract the information in this "
    "section as tight bullet points, without dropping any important "
    "definition, formula or distinction. No preamble, no commentary, no "
    "information that is not in the text."
)


def _system_prompt(language: str, partial: bool) -> str:
    if partial:
        return SYSTEM_PROMPT_PARTIAL_TR if language == "tr" else SYSTEM_PROMPT_PARTIAL_EN
    return SYSTEM_PROMPT_SUMMARY_TR if language == "tr" else SYSTEM_PROMPT_SUMMARY_EN


def group_chunks(chunks: list[str], max_chars: int | None = None) -> list[list[str]]:
    """Parçaları, her biri `max_chars` sınırını aşmayan ardışık gruplara böler.

    SIRA KORUNUR: özet için parçaların doküman sırasında kalması şart (aksi
    halde "önce sonuç, sonra tanım" gibi bozuk bir anlatım çıkar) — bu yüzden
    `/ask`'teki gibi alakaya göre yeniden sıralama YAPILMIYOR.

    Tek bir parça sınırdan büyükse kendi başına bir grup olur (bölmüyoruz:
    chunking zaten ingestion aşamasında yapıldı, burada tekrar bölmek cümle
    ortasından kesme riski taşır).
    """
    if max_chars is None:
        max_chars = settings.summary_group_max_chars

    groups: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    for chunk in chunks:
        if current and current_len + len(chunk) > max_chars:
            groups.append(current)
            current = []
            current_len = 0
        current.append(chunk)
        current_len += len(chunk)

    if current:
        groups.append(current)
    return groups


def _build_messages(texts: list[str], language: str, partial: bool) -> list[dict]:
    body = "\n\n---\n\n".join(texts)
    label = "Metin" if language == "tr" else "Text"
    instruction = (
        "Yukarıdaki metni özetle." if language == "tr" else "Summarize the text above."
    )
    return [
        {"role": "system", "content": _system_prompt(language, partial)},
        {"role": "user", "content": f"{label}:\n{body}\n\n{instruction}"},
    ]


def _client_and_model():
    manager = _get_manager()
    model_id = _get_model_id()
    from openai import OpenAI

    return OpenAI(base_url=manager.endpoint, api_key=manager.api_key), model_id


# NOT: kapasite/ölümcül GPU hatası ayrımı `services/llm.py`'de yapılıyor
# (bkz. `is_capacity_error`, `raise_if_gpu_context_lost`, `GpuContextLost`) —
# aynı ayrım `/ask` ve `/quiz` için de gerekli olduğundan orada ortak tutuldu.
# Buradaki `_is_capacity_error` yalnızca geriye dönük uyumluluk/testler için
# duruyor.
_is_capacity_error = is_capacity_error


def _summarize_once(texts: list[str], language: str, partial: bool, max_tokens: int) -> str:
    client, model_id = _client_and_model()
    response = _create_chat_completion(
        client,
        model=model_id,
        messages=_build_messages(texts, language, partial),
        # Özette yaratıcılık istemiyoruz; düşük sıcaklık metne sadık kalmayı
        # ve uydurmayı azaltmayı sağlıyor.
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return strip_think(response.choices[0].message.content or "")


def _stream_once(texts: list[str], language: str, partial: bool, max_tokens: int):
    client, model_id = _client_and_model()
    stream = _create_chat_completion(
        client,
        model=model_id,
        messages=_build_messages(texts, language, partial),
        temperature=0.2,
        max_tokens=max_tokens,
        stream=True,
    )
    yield from _stream_with_warmup(_stream_and_strip(stream))


# Bir metni ikiye bölmenin anlamlı olduğu asgari uzunluk. Bunun altında
# bölmeye devam etmek, modele anlamsız derecede küçük parçalar göndermek
# demek — kapasite sorunu bu boyutta zaten metinden kaynaklanmıyordur.
_MIN_SPLITTABLE_CHARS = 400


def _split_text(text: str) -> list[str] | None:
    """Tek bir metni, ortaya en yakın boşluktan ikiye böler.

    NEDEN GEREKLİ: kapasite hatasında "parça listesini ikiye böl" yaklaşımı,
    listede TEK parça varsa işe yaramıyor. Bu gerçekte sık karşılaşılan bir
    durum: `chunk_size` küçültülmeden ÖNCE yüklenmiş dosyaların parçaları çok
    daha büyük (~3000+ karakter), yani tek bir parça bile bir grubu tek
    başına dolduruyor. Gözlemlenen arıza tam olarak buydu — fallback
    "bölecek bir şey yok" deyip hatayı yükseltiyordu.

    Kelime ortasından kesmemek için ortaya en yakın boşluk aranıyor.
    """
    if len(text) < _MIN_SPLITTABLE_CHARS:
        return None
    middle = len(text) // 2
    split_at = text.rfind(" ", 0, middle)
    if split_at <= 0:
        split_at = middle
    left, right = text[:split_at].strip(), text[split_at:].strip()
    if not left or not right:
        return None
    return [left, right]


def _split_for_retry(texts: list[str]) -> tuple[list[str], list[str]] | None:
    """Kapasite hatasından sonra tekrar denenecek iki yarıyı üretir.

    Birden fazla parça varsa listeyi ikiye bölmek yeterli; tek parça varsa
    METNİN KENDİSİNİ bölüyoruz (bkz. `_split_text`). Daha fazla bölünemiyorsa
    `None` döner ve çağıran taraf hatayı yükseltir.
    """
    if len(texts) >= 2:
        middle = len(texts) // 2
        return texts[:middle], texts[middle:]
    if len(texts) == 1:
        halves = _split_text(texts[0])
        if halves is not None:
            return [halves[0]], [halves[1]]
    return None


def _summarize_with_fallback(
    texts: list[str], language: str, partial: bool, max_tokens: int
) -> str:
    """`_summarize_once`, ama model kapasiteye çarparsa girdiyi İKİYE BÖLÜP
    tekrar dener (özyinelemeli).

    Neden gerekli: özet context'inin ne kadar büyük olabileceği kullanıcının
    donanımına bağlı ve bunu önceden bilemiyoruz. Sabit bir sınır seçmek ya
    güçlü makinelerde gereksiz çok tura yol açar ya da zayıf makinelerde
    çuvallar. Bunun yerine iyimser başlayıp, çarparsak küçülüyoruz — sonuçta
    her makinede çalışan ama hiçbirinde gereksiz yavaşlamayan bir davranış.

    Bölme hem parça listesi hem de TEK BİR PARÇANIN METNİ üzerinde çalışıyor
    (bkz. `_split_for_retry`) — aksi halde tek ve büyük bir parça (eski
    `chunk_size` ile yüklenmiş dosyalarda tipik) fallback'i baştan devre dışı
    bırakıyordu. Artık bölünemeyecek kadar küçüldüyse hata olduğu gibi
    yükseliyor ki kullanıcı gerçek nedeni görsün.
    """
    try:
        return _summarize_once(texts, language, partial, max_tokens)
    except Exception as exc:
        # ÖNCE ölümcül kontrol: CUDA bağlamı bozulduysa bölerek tekrar
        # denemek ANLAMSIZ — bağlam zehirlendiği için her boyutta aynı hata
        # geliyor (gözlemlenen: 3145 -> 1562 -> 770 -> 385 karakter, dördü de
        # başarısız). Kullanıcıyı bekletmek yerine hemen anlaşılır bir hataya
        # çeviriyoruz.
        raise_if_gpu_context_lost(exc)
        if not is_capacity_error(exc):
            raise
        pieces = _split_for_retry(texts)
        if pieces is None:
            raise
        logger.warning(
            "Model kapasitesi aşıldı (%d parça, %d karakter); bölünüp tekrar deneniyor",
            len(texts),
            sum(len(t) for t in texts),
        )

    left, right = pieces
    halves = [
        _summarize_with_fallback(left, language, partial=True, max_tokens=max_tokens),
        _summarize_with_fallback(right, language, partial=True, max_tokens=max_tokens),
    ]
    combined = [h for h in halves if h.strip()]
    if not combined:
        raise ValueError("Model bu bölüm için özet üretemedi")

    if partial:
        # Ara özet zaten; iki yarıyı birleştirmek yeterli, ikinci bir
        # damıtma turuna gerek yok.
        return "\n\n".join(combined)
    # Nihai özet isteniyordu: iki yarı özetini tek metne indirge (bu adımın
    # context'i çok daha küçük olduğu için genelde sorunsuz geçiyor).
    return _summarize_with_fallback(combined, language, partial=False, max_tokens=max_tokens)


def generate_summary(chunks: list[str], language: str = "tr") -> str:
    """Dokümanın tamamı için tek seferde (streaming olmayan) özet üretir.

    Programatik kullanım ve testler için; arayüz `generate_summary_stream`
    kullanıyor.
    """
    if not chunks:
        raise ValueError("Özetlenecek içerik yok")

    groups = group_chunks(chunks)
    if len(groups) == 1:
        return _summarize_with_fallback(groups[0], language, partial=False,
                                        max_tokens=settings.summary_max_tokens)

    partials = [
        _summarize_with_fallback(group, language, partial=True,
                                 max_tokens=settings.summary_partial_max_tokens)
        for group in groups
    ]
    return _summarize_with_fallback(
        [p for p in partials if p.strip()],
        language,
        partial=False,
        max_tokens=settings.summary_max_tokens,
    )


def _stream_final(texts: list[str], language: str):
    """Nihai özeti akıtır; model kapasiteye çarparsa akışı bırakıp
    bölerek-özetleyen (streaming olmayan) yola düşer.

    Kapasite hatası istek AÇILIRKEN geldiği için (henüz tek bir token bile
    yayınlanmamış olur) bu geçiş kullanıcıya yarım bir metin göstermeden
    yapılabiliyor — sonuç tek parça hâlinde gelir, canlı akmaz, ama gelir.
    """
    produced = False
    try:
        for piece in _stream_once(texts, language, partial=False,
                                  max_tokens=settings.summary_max_tokens):
            produced = True
            yield ("token", piece)
        return
    except Exception as exc:
        raise_if_gpu_context_lost(exc)
        if produced or not is_capacity_error(exc):
            raise
        logger.warning("Nihai özet akışı kapasiteye takıldı; bölerek yeniden deneniyor")

    yield ("progress", "yeniden deneniyor")
    yield ("token", _summarize_with_fallback(
        texts, language, partial=False, max_tokens=settings.summary_max_tokens
    ))


def generate_summary_stream(chunks: list[str], language: str = "tr"):
    """`generate_summary`'nin streaming karşılığı.

    İki tip olay üretir (çağıran taraf bunları SSE'ye çevirir, bkz.
    api/summary.py):

        ("progress", "3/7")  -> map adımında kaçıncı bölümün özetlendiği
        ("token", "...")     -> NİHAİ özetin metin parçaları

    Map adımının ara çıktıları KULLANICIYA GÖSTERİLMEZ (onlar sıkıştırılmış
    ham notlar, okunacak bir şey değil); yalnızca ilerleme bilgisi gönderilir.
    Kısa dokümanlarda (tek grup) map adımı hiç çalışmaz ve nihai özet doğrudan
    akmaya başlar.
    """
    if not chunks:
        raise ValueError("Özetlenecek içerik yok")

    groups = group_chunks(chunks)

    if len(groups) == 1:
        yield from _stream_final(groups[0], language)
        return

    partials: list[str] = []
    for index, group in enumerate(groups, start=1):
        yield ("progress", f"{index}/{len(groups)}")
        partial = _summarize_with_fallback(
            group, language, partial=True, max_tokens=settings.summary_partial_max_tokens
        )
        if partial.strip():
            partials.append(partial)
        else:
            logger.warning("Özet map adımı boş döndü (bölüm %d/%d)", index, len(groups))

    if not partials:
        raise ValueError("Model hiçbir bölüm için özet üretemedi")

    yield ("progress", "birleştiriliyor")
    yield from _stream_final(partials, language)
