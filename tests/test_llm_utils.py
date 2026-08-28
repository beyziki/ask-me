"""llm.py'deki saf (network/Foundry gerektirmeyen) yardımcı fonksiyonlar için
birim testleri: prompt oluşturma, <think> bloğu temizleme, tekrar/boş cevap
korumaları ve modele göre değişen davranışlar.
"""
import httpx
import pytest

from backend.app.core.config import settings
from backend.app.services.llm import (
    TRANSPORT_ERRORS,
    _NO_THINK_INSTRUCTION_TR,
    _create_chat_completion,
    _is_blank,
    _looks_degenerate,
    _looks_too_short,
    _RepetitionGuard,
    _stream_and_strip,
    _stream_with_warmup,
    _ThinkStreamStripper,
    build_prompt,
    strip_think,
    SYSTEM_PROMPT_NO_CONTEXT_TR,
)


@pytest.fixture(autouse=True)
def _plain_model(monkeypatch):
    """Testlerin varsayılanı: DÜŞÜNMEYEN (instruct) model.

    `build_prompt`/`_create_chat_completion` artık modele göre farklı
    davrandığı için (bkz. config.py:model_has_thinking), testler bunu
    ortamdaki `.env`'e bırakmak yerine açıkça sabitliyor. Thinking
    davranışını doğrulayan testler bu değeri kendi içinde geçersiz kılıyor.
    """
    monkeypatch.setattr(settings, "model_thinking", False)


def _feed_all(chunks: list[str]) -> str:
    """Bir dizi delta'yı `_ThinkStreamStripper`'a besleyip tüm çıktıyı birleştirir."""
    stripper = _ThinkStreamStripper()
    out = "".join(stripper.feed(c) for c in chunks)
    out += stripper.flush()
    return out


class _FakeDelta:
    def __init__(self, content: str | None):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None):
        self.delta = _FakeDelta(content)


class _FakeEvent:
    """OpenAI streaming chunk'ının `_stream_and_strip`'in kullandığı kısmını
    (yalnızca `.choices[0].delta.content`) taklit eden sahte nesne."""

    def __init__(self, content: str | None):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """`client.chat.completions` yerine geçen sahte nesne; `_create_chat_completion`'ın
    önce `extra_body` ile (Qwen3'ün `enable_thinking=False` anahtarı), o
    reddedilirse `extra_body` olmadan tekrar denediğini doğrulamak için
    kullanılıyor. `reject_extra_body=True` olduğunda `extra_body` içeren
    çağrıyı gerçek `openai.BadRequestError`'a çok benzer şekilde reddeder."""

    def __init__(self, reject_extra_body: bool):
        self.reject_extra_body = reject_extra_body
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_extra_body and "extra_body" in kwargs:
            import httpx
            from openai import BadRequestError

            response = httpx.Response(status_code=400, request=httpx.Request("POST", "http://fake"))
            raise BadRequestError("unknown field: extra_body", response=response, body=None)
        return "ok"


class _FakeClient:
    """`OpenAI` istemcisinin `_create_chat_completion`'ın kullandığı
    `.chat.completions.create(...)` yolunu taklit eden sahte nesne."""

    def __init__(self, reject_extra_body: bool):
        self.chat = type("_Chat", (), {})()
        self.chat.completions = _FakeCompletions(reject_extra_body)


class _FakeEmptyChoicesEvent:
    """Bazı sunucuların (gözlemlenen: Foundry Local) ara sıra gönderdiği,
    `choices` listesi boş olan chunk'ı taklit eder."""

    def __init__(self):
        self.choices = []


def test_strip_think_removes_block():
    raw = "<think>bir sürü akıl yürütme burada...</think>Asıl cevap burada."
    assert strip_think(raw) == "Asıl cevap burada."


def test_strip_think_noop_when_no_block():
    raw = "Sadece düz bir cevap, hiç think bloğu yok."
    assert strip_think(raw) == raw


def test_strip_think_handles_multiline_and_multiple_blocks():
    raw = "<think>\nsatır1\nsatır2\n</think>\n\nCevap 1. <think>ikinci blok</think>Cevap 2."
    result = strip_think(raw)
    assert "<think>" not in result
    assert "Cevap 1." in result
    assert "Cevap 2." in result


def test_build_prompt_includes_question_and_context():
    messages = build_prompt("Turing makinesi nedir?", ["kaynak parçası"], "tr")
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Turing makinesi nedir?" in messages[1]["content"]
    assert "kaynak parçası" in messages[1]["content"]


def test_build_prompt_appends_no_think_switch_only_for_thinking_models(monkeypatch):
    # Thinking modeli (ör. qwen3-8b): `/no_think` soft-switch'i ve akıl
    # yürütmeyi bastıran sistem talimatı eklenmeli.
    monkeypatch.setattr(settings, "model_thinking", True)
    thinking = build_prompt("soru", ["kaynak"], "tr")
    assert "/no_think" in thinking[1]["content"]
    assert "<think>" in thinking[0]["content"]

    # Düşünmeyen (instruct) model: bastırılacak akıl yürütme yok, bu yüzden
    # `/no_think` metni (cevaba sızma riskiyle birlikte) eklenmemeli.
    monkeypatch.setattr(settings, "model_thinking", False)
    plain = build_prompt("soru", ["kaynak"], "tr")
    assert "/no_think" not in plain[1]["content"]
    assert "<think>" not in plain[0]["content"]


def test_build_prompt_uses_english_system_prompt_for_en():
    tr_messages = build_prompt("soru", ["kaynak"], "tr")
    en_messages = build_prompt("question", ["source"], "en")
    assert tr_messages[0]["content"] != en_messages[0]["content"]


def test_build_prompt_switches_to_general_knowledge_when_no_context():
    # Hybrid RAG yeterince ilgili parça bulamadığında (bkz.
    # api/ask.py:_retrieve) alakasız context HİÇ gönderilmemeli; model,
    # cevabın yüklenen dosyalardan gelmediğini belirterek genel bilgiyle
    # cevap vermeye yönlendirilmeli.
    messages = build_prompt("soru", ["alakasız parça"], "tr", has_context=False)
    assert "alakasız parça" not in messages[1]["content"]
    assert messages[0]["content"] == SYSTEM_PROMPT_NO_CONTEXT_TR


def test_build_prompt_treats_empty_context_as_no_context():
    # `has_context` verilmese bile boş context aynı moda düşmeli.
    messages = build_prompt("soru", [], "tr")
    assert messages[0]["content"] == SYSTEM_PROMPT_NO_CONTEXT_TR


def test_build_prompt_no_context_still_suppresses_thinking_when_needed(monkeypatch):
    monkeypatch.setattr(settings, "model_thinking", True)
    messages = build_prompt("soru", [], "tr")
    assert messages[0]["content"] == SYSTEM_PROMPT_NO_CONTEXT_TR + _NO_THINK_INSTRUCTION_TR


def test_think_stream_stripper_removes_whole_block_in_one_chunk():
    assert _feed_all(["<think>akıl yürütme</think>Asıl cevap."]) == "Asıl cevap."


def test_think_stream_stripper_handles_tag_split_across_chunks():
    # Gerçek streaming'de <think> etiketinin kendisi bile birden fazla
    # delta'ya bölünebilir (ör. "<thi" + "nk>"); bu test o senaryoyu simüle
    # ediyor.
    chunks = ["<thi", "nk>gizli akıl yürütme", "</th", "ink>", "Görünen cevap."]
    assert _feed_all(chunks) == "Görünen cevap."


def test_think_stream_stripper_passes_through_text_with_no_think_block():
    chunks = ["Sadece ", "düz ", "bir ", "cevap."]
    assert _feed_all(chunks) == "Sadece düz bir cevap."


def test_think_stream_stripper_handles_multiple_blocks():
    chunks = ["<think>ilk</think>", "Cevap 1. ", "<think>ikinci</think>", "Cevap 2."]
    assert _feed_all(chunks) == "Cevap 1. Cevap 2."


def test_think_stream_stripper_drops_unterminated_trailing_think_block():
    # Akış <think> içindeyken kesilirse (ör. bağlantı koptu), yarım/kesik
    # akıl yürütme metnini kullanıcıya göstermek yerine sessizce düşürüyoruz.
    chunks = ["Görünen kısım. ", "<think>bitmemiş akıl yürütme..."]
    assert _feed_all(chunks) == "Görünen kısım. "


def test_stream_and_strip_yields_text_and_strips_think():
    events = [_FakeEvent("<think>gizli</think>"), _FakeEvent("Merhaba "), _FakeEvent("dünya.")]
    assert "".join(_stream_and_strip(events)) == "Merhaba dünya."


def test_stream_and_strip_ignores_empty_deltas():
    # Bazı chunk'lar (ör. yalnızca finish_reason taşıyanlar) boş/None
    # `delta.content` içerebilir; bunlar sessizce atlanmalı.
    events = [_FakeEvent(None), _FakeEvent("Cevap"), _FakeEvent("")]
    assert "".join(_stream_and_strip(events)) == "Cevap"


def test_stream_and_strip_ignores_events_with_empty_choices():
    # Gözlemlenen bir Foundry Local davranışı: OpenAI'ın gerçek API'sinde
    # `choices` yalnızca istenirse (stream_options.include_usage) ve yalnızca
    # sonda boş gelir; Foundry Local bunu istenmeden de gönderebiliyor.
    # `event.choices[0]` bunu kontrol etmeden indekslemek IndexError'a yol
    # açıyordu (gözlemlenen gerçek çökme) — burada sessizce atlanmalı.
    events = [_FakeEvent("Merhaba "), _FakeEmptyChoicesEvent(), _FakeEvent("dünya.")]
    assert "".join(_stream_and_strip(events)) == "Merhaba dünya."


def test_stream_and_strip_swallows_transport_error_after_content_received():
    # Gözlemlenen Foundry Local davranışı: üretim bittikten hemen sonra
    # bağlantı chunked-encoding'in kapanış işaretini göndermeden kesiliyor.
    # İçerik zaten akmışsa bunu bir hata olarak değil, akışın (biraz
    # düzensiz) bitişi olarak ele alıyoruz; elimizdeki metin kaybolmuyor.
    def events():
        yield _FakeEvent("Merhaba ")
        yield _FakeEvent("dünya")
        raise httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body (incomplete chunked read)"
        )

    assert "".join(_stream_and_strip(events())) == "Merhaba dünya"


def test_stream_and_strip_reraises_transport_error_when_no_content_received():
    # Bağlantı hiç içerik gelmeden koparsa bu gerçek bir sorunun işareti;
    # burada sessizce yutmuyoruz, çağıran taraf (generate_answer_stream)
    # bunu kullanıcıya gösterilebilir bir hataya çevirebilsin diye.
    def events():
        raise httpx.RemoteProtocolError("connection closed")
        yield  # pragma: no cover - generator olması için gerekli, hiç çalışmaz

    with pytest.raises(httpx.TransportError):
        list(_stream_and_strip(events()))


def test_repetition_guard_ignores_normal_prose():
    guard = _RepetitionGuard()
    words = "Bu gayet normal ve çeşitli kelimelerden oluşan bir cümle örneğidir".split(" ")
    assert not any(guard.feed(w + " ") for w in words)


def test_repetition_guard_triggers_on_ten_consecutive_repeats():
    guard = _RepetitionGuard(max_consecutive_repeats=10)
    # 9 tekrar henüz eşiği geçmemeli.
    assert not guard.feed("Otonom " * 9)
    # 10.'da True dönmeli.
    assert guard.feed("Otonom ")


def test_repetition_guard_resets_run_on_different_word():
    guard = _RepetitionGuard(max_consecutive_repeats=10)
    # 9 kez "Otonom", sonra farklı bir kelime -> sayaç sıfırlanmalı, tetiklenmemeli.
    assert not guard.feed("Otonom " * 9 + "Motor ")
    # Ardından yalnızca 5 kez daha "Otonom" gelirse eşiğe (10) ulaşılmaz.
    assert not guard.feed("Otonom " * 5)


def test_repetition_guard_counts_words_split_across_feed_calls():
    # "Otonom" kelimesi boşluğa kadar tamponda tutulur; parça parça (delta
    # delta) gelen streaming'i simüle ediyoruz.
    guard = _RepetitionGuard(max_consecutive_repeats=3)
    triggered = False
    for piece in ["Oto", "nom ", "Otonom", " ", "Oton", "om "]:
        triggered = triggered or guard.feed(piece)
    assert triggered


def test_repetition_guard_matches_observed_degenerate_output():
    # Kullanıcının gerçekte gördüğü bozuk çıktıyı (kısaltılmış) simüle ediyor:
    # önce "Otonom" onlarca kez, sonra tek harf "O" onlarca kez. Her kelimeyi
    # sonuna boşluk ekleyerek besliyoruz (`feed`, kelime sınırlarını
    # boşluklardan tanıyor).
    degenerate_words = ["Otonom"] * 15 + ["O"] * 15
    guard = _RepetitionGuard()
    assert any(guard.feed(word + " ") for word in degenerate_words)


def test_repetition_guard_catches_observed_phrase_loop():
    """Kullanıcının gerçekte gördüğü arıza: AYNI KELİME hiç art arda
    gelmiyor, altı kelimelik bir öbek dönüp duruyor. Eski koruma (yalnızca
    ardışık aynı kelime) bunu hiç yakalamıyordu."""
    phrase = "çapkalarda kullanma ve kullanma için kullanma, "
    guard = _RepetitionGuard()
    assert any(guard.feed(phrase) for _ in range(6))


def test_repetition_guard_catches_two_word_loop():
    guard = _RepetitionGuard()
    assert any(guard.feed("kullanma için ") for _ in range(8))


def test_repetition_guard_phrase_loop_needs_enough_repeats():
    # Bir öbeğin üç kez geçmesi (ör. bir listede) tek başına arıza değil.
    guard = _RepetitionGuard(min_phrase_repeats=4)
    assert not guard.feed("bir iki " * 3 + "üç dört beş ")


def test_repetition_guard_ignores_punctuation_and_case_in_phrase_loop():
    # Döngüdeki öbek her turda birebir aynı yazılmayabiliyor.
    guard = _RepetitionGuard()
    variants = ["Kullanma için ", "kullanma, için ", "KULLANMA için ", "kullanma için "]
    assert any(guard.feed(v) for v in variants)


def test_repetition_guard_ignores_long_natural_prose():
    """Yanlış pozitif kontrolü: uzun ve tekrarsız bir metin tetiklememeli."""
    text = (
        "Turing makinesi, sonsuz uzunlukta bir bant üzerinde okuma ve yazma "
        "yapabilen soyut bir hesaplama modelidir. Alan Turing bu modeli 1936 "
        "yılında tanımladı. Model, bir durum kümesi, bir geçiş fonksiyonu ve "
        "bant üzerinde hareket eden bir okuma yazma kafasından oluşur. "
        "Hesaplanabilirlik kuramının temel taşlarından biri olarak kabul edilir "
        "ve modern bilgisayarların kuramsal atası sayılır. "
    )
    guard = _RepetitionGuard()
    assert not any(guard.feed(word + " ") for word in text.split())


def test_looks_degenerate_true_for_phrase_loop():
    text = "Cevap: " + "kullanma ve kullanma için " * 5
    assert _looks_degenerate(text)


def test_looks_degenerate_true_for_repeated_word():
    text = "Cevap: " + "Otonom " * 12
    assert _looks_degenerate(text)


def test_looks_degenerate_false_for_normal_answer():
    text = (
        "Turing makinesi, sonsuz bir bant üzerinde okuma/yazma yapabilen "
        "soyut bir hesaplama modelidir. Alan Turing tarafından 1936'da "
        "tanımlanmıştır."
    )
    assert not _looks_degenerate(text)


def test_is_blank_true_for_empty_or_whitespace_only():
    assert _is_blank("")
    assert _is_blank("   \n\t  ")


def test_is_blank_false_for_real_text():
    assert not _is_blank("Turing makinesi nedir?")


def test_stream_and_strip_yields_nothing_when_stuck_inside_unterminated_think():
    # Gözlemlenen gerçek arıza: `/no_think`'e rağmen model akıl yürütmeye
    # devam ediyor ve `<think>` bloğunu hiç kapatmadan token bütçesi
    # bitiyor. `_ThinkStreamStripper`, kapanmamış bir think bloğunun
    # içeriğini kasıtlı olarak düşürdüğü için (bkz. flush) bu durumda hiçbir
    # parça yayınlanmamalı — `generate_answer_stream` bunu `EmptyAnswer` ile
    # açık bir hataya çevirir (bkz. test_llm_utils.py'nin üstündeki not).
    events = [
        _FakeEvent("<think>uzun akıl yürütme başlıyor"),
        _FakeEvent(" ... token sınırına kadar bitmeden sürüyor"),
    ]
    assert list(_stream_and_strip(events)) == []


def test_looks_too_short_true_for_single_punctuation_char():
    # Gözlemlenen gerçek arıza: model `<think>` bloğunu kapatıp görünür bir
    # cevaba geçiyor ama geriye tek bir "." kalıyor. `_is_blank` bunu
    # yakalamıyor (boş değil), `_looks_too_short` yakalamalı.
    assert _looks_too_short(".")
    assert _looks_too_short("  . ")
    assert _looks_too_short("ab")


def test_looks_too_short_false_for_empty_string():
    # Tamamen boş metin `_is_blank`'in sorumluluğunda; `_looks_too_short`
    # onunla çakışmamalı (ikisi birlikte `or` ile kullanılıyor, bkz. llm.py).
    assert not _looks_too_short("")
    assert not _looks_too_short("   ")


def test_looks_too_short_false_for_real_answer():
    assert not _looks_too_short("Turing makinesi nedir?")
    assert not _looks_too_short("Evet")


def test_stream_with_warmup_withholds_output_below_threshold():
    # Akışın TAMAMI eşiğin (4 karakter) altında bitiyor -- gerçek gözlemlenen
    # arıza: model yalnızca "." üretip duruyor. Hiçbir şey yayınlanmamalı.
    assert list(_stream_with_warmup(iter(["."]))) == []


def test_stream_with_warmup_buffers_until_threshold_then_streams_live():
    # İlk birkaç parça eşiğin altında birikiyor, eşik aşılınca hepsi birden
    # (tek bir "toplu" parça olarak) yayınlanıyor, sonrasında normal parça
    # parça akış devam ediyor.
    pieces = list(_stream_with_warmup(iter(["T", "u", "r", "i", "ng makinesi ", "nedir?"])))
    assert "".join(pieces) == "Turing makinesi nedir?"
    # İlk 4 karakter tek seferde (buffer flush) gelmeli, sonrası ayrı ayrı.
    assert pieces[0] == "Turi"
    assert pieces[1:] == ["ng makinesi ", "nedir?"]


def test_stream_with_warmup_passes_through_long_first_piece_immediately():
    # Tek bir parça zaten eşiği aşıyorsa (tipik gerçek akış), gecikme
    # olmadan hemen yayınlanmalı.
    pieces = list(_stream_with_warmup(iter(["Turing makinesi sonsuz bir bant üzerinde..."])))
    assert pieces == ["Turing makinesi sonsuz bir bant üzerinde..."]


def test_create_chat_completion_skips_extra_body_for_non_thinking_model():
    # Düşünmeyen (instruct) modelde bastırılacak akıl yürütme yok: ekstra
    # alanı hiç göndermeyip tek çağrıda bitirmeli (gereksiz bir 400 + tekrar
    # denemesi riski de ortadan kalkıyor).
    client = _FakeClient(reject_extra_body=False)
    result = _create_chat_completion(client, model="m", messages=[])
    assert result == "ok"
    assert len(client.chat.completions.calls) == 1
    assert "extra_body" not in client.chat.completions.calls[0]


def test_create_chat_completion_uses_enable_thinking_when_accepted(monkeypatch):
    # Thinking modelinde, altındaki motor `extra_body`'yi (Qwen3'ün resmi
    # `enable_thinking=False` anahtarı) kabul ederse tek çağrı yeterli.
    monkeypatch.setattr(settings, "model_thinking", True)
    client = _FakeClient(reject_extra_body=False)
    result = _create_chat_completion(client, model="m", messages=[])
    assert result == "ok"
    assert len(client.chat.completions.calls) == 1
    assert client.chat.completions.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_create_chat_completion_falls_back_when_extra_body_rejected(monkeypatch):
    # Motor bu alanı tanımıyorsa (400 Bad Request) sessizce, aynı isteği
    # `extra_body` olmadan tekrar deneyip eski/güvenli davranışa dönmeli —
    # kullanıcı bir hata görmemeli, sadece `enable_thinking` avantajından
    # yararlanamamış olur.
    monkeypatch.setattr(settings, "model_thinking", True)
    client = _FakeClient(reject_extra_body=True)
    result = _create_chat_completion(client, model="m", messages=[])
    assert result == "ok"
    assert len(client.chat.completions.calls) == 2
    assert "extra_body" in client.chat.completions.calls[0]
    assert "extra_body" not in client.chat.completions.calls[1]


def test_answer_token_budgets_depend_on_model_type(monkeypatch):
    # Düşünmeyen modelde tek ve küçük bir bütçe yeterli: üretilen her token
    # doğrudan görünür cevap, "bütçeyi akıl yürütmede tüketme" arızası yok.
    monkeypatch.setattr(settings, "model_thinking", False)
    assert settings.answer_token_budgets == [settings.answer_max_tokens]

    # Thinking modelinde eski iki kademeli eskalasyon korunuyor.
    monkeypatch.setattr(settings, "model_thinking", True)
    budgets = settings.answer_token_budgets
    assert len(budgets) == 2
    assert budgets[1] > budgets[0]


def test_model_has_thinking_inferred_from_alias(monkeypatch):
    # `.env`'de açık bir MODEL_THINKING yoksa alias'tan çıkarılmalı.
    monkeypatch.setattr(settings, "model_thinking", None)
    monkeypatch.setattr(settings, "foundry_model_alias", "qwen3-8b")
    assert settings.model_has_thinking
    monkeypatch.setattr(settings, "foundry_model_alias", "qwen2.5-7b-instruct")
    assert not settings.model_has_thinking
    monkeypatch.setattr(settings, "foundry_model_alias", "phi-4-mini-instruct")
    assert not settings.model_has_thinking


# --- Taşıma (transport) hatası sınıfları ----------------------------------
# Gözlemlenen gerçek arıza: `openai` 3.x, `httpx` yerine `httpx2` kullanıyor.
# Bu iki paketin istisna sınıfları BİRBİRİNDEN BAĞIMSIZ
# (`httpx2.RemoteProtocolError`, `httpx.TransportError`'ın alt sınıfı değil),
# bu yüzden yalnızca `httpx.TransportError` yakalayan kod Foundry Local'in
# bilinen "bağlantıyı yarıda kesme" davranışını kaçırıyor ve /quiz/stream ham
# traceback'le çöküyordu.


def test_transport_errors_covers_installed_http_libraries():
    assert TRANSPORT_ERRORS, "en az bir taşıma hatası sınıfı bulunmalı"
    assert all(isinstance(t, type) for t in TRANSPORT_ERRORS)
    assert all(issubclass(t, BaseException) for t in TRANSPORT_ERRORS)


def test_transport_errors_includes_httpx_when_available():
    assert httpx.TransportError in TRANSPORT_ERRORS


def test_transport_errors_includes_httpx2_when_available():
    # httpx2 kurulu DEĞİLSE bu test anlamsız; kuruluysa mutlaka kapsanmalı.
    httpx2 = pytest.importorskip("httpx2")
    assert httpx2.TransportError in TRANSPORT_ERRORS


def test_stream_and_strip_swallows_httpx2_error_after_content(monkeypatch):
    # httpx2 kuruluysa, onun hatası da tıpkı httpx'inki gibi (içerik akmışsa)
    # sessizce yutulmalı — kullanıcı cevabını kaybetmemeli.
    httpx2 = pytest.importorskip("httpx2")

    def events():
        yield _FakeEvent("Merhaba ")
        yield _FakeEvent("dünya")
        raise httpx2.RemoteProtocolError("incomplete chunked read")

    assert "".join(_stream_and_strip(events())) == "Merhaba dünya"


def test_repetition_guard_phrase_rule_does_not_lower_single_word_threshold():
    """Tek kelimenin tekrarı, öbek kuralı üzerinden erken tetiklenmemeli:
    o durumun kendi (daha yüksek) eşiği var."""
    guard = _RepetitionGuard(max_consecutive_repeats=10)
    assert not guard.feed("Otonom " * 9)
    assert guard.feed("Otonom ")
