"""quiz.py'deki saf (network/Foundry gerektirmeyen) yardımcı fonksiyonlar
için birim testleri: JSON ayrıştırma ve max_tokens hesaplama.
"""
import pytest

from backend.app.services.quiz import (
    _QUIZ_MAX_TOKENS_CEILING,
    _parse_quiz_json,
    _quiz_max_tokens,
)


def test_parse_quiz_json_handles_plain_json():
    raw = '[{"question": "2+2?", "options": null, "answer": "4"}]'
    assert _parse_quiz_json(raw) == [{"question": "2+2?", "options": None, "answer": "4"}]


def test_parse_quiz_json_handles_code_fenced_json():
    # Model bazen ```json ... ``` kod bloğu içinde JSON döndürüyor.
    raw = '```json\n[{"question": "Soru?", "options": null, "answer": "Cevap"}]\n```'
    assert _parse_quiz_json(raw) == [{"question": "Soru?", "options": None, "answer": "Cevap"}]


def test_parse_quiz_json_raises_on_truly_invalid_json():
    with pytest.raises(Exception):
        _parse_quiz_json("bu hiç JSON değil")


# --- Modelin "sadece JSON döndür" talimatına uymadığı durumlar -------------
# Gözlemlenen hata: "Expecting value: line 1 column 1 (char 0)" — JSON'ın
# önüne/arkasına düz metin eklenince tüm quiz kayboluyordu.


def test_parse_quiz_json_ignores_text_before_and_after_json():
    raw = (
        "İşte istediğin çalışma soruları:\n\n"
        '[{"question": "Turing makinesi nedir?", "options": null, "answer": "Soyut bir model"}]'
        "\n\nUmarım yardımcı olur!"
    )
    assert _parse_quiz_json(raw) == [
        {"question": "Turing makinesi nedir?", "options": None, "answer": "Soyut bir model"}
    ]


def test_parse_quiz_json_accepts_object_wrapper():
    # Model bazen düz dizi yerine onu saran bir nesne döndürüyor.
    raw = '{"questions": [{"question": "S?", "options": null, "answer": "C"}]}'
    assert _parse_quiz_json(raw) == [{"question": "S?", "options": None, "answer": "C"}]


def test_parse_quiz_json_accepts_single_question_object():
    raw = '{"question": "Tek soru?", "options": null, "answer": "Cevap"}'
    assert _parse_quiz_json(raw) == [
        {"question": "Tek soru?", "options": None, "answer": "Cevap"}
    ]


def test_parse_quiz_json_handles_brackets_inside_question_text():
    # Naif bir `find('[')` + `rfind(']')` bu metinde yanlış yeri keserdi;
    # parantez sayımı string literal içindekileri saymamalı.
    raw = (
        "Sorular:\n"
        '[{"question": "[Kaynak 1] neyi anlatıyor?", "options": ["a]", "b["], "answer": "a]"}]'
    )
    parsed = _parse_quiz_json(raw)
    assert parsed[0]["question"] == "[Kaynak 1] neyi anlatıyor?"
    assert parsed[0]["options"] == ["a]", "b["]


def test_parse_quiz_json_handles_fence_with_surrounding_text():
    raw = 'Tabii!\n```json\n[{"question": "S?", "options": null, "answer": "C"}]\n```\nKolay gelsin.'
    assert _parse_quiz_json(raw) == [{"question": "S?", "options": None, "answer": "C"}]


def test_parse_quiz_json_gives_clear_error_on_empty_output():
    # Eskiden bu da "Expecting value: line 1 column 1" veriyordu; artık ne
    # olduğu anlaşılıyor.
    with pytest.raises(ValueError, match="boş yanıt"):
        _parse_quiz_json("   \n  ")


def test_parse_quiz_json_error_includes_model_output_for_diagnosis():
    # Hata mesajı modelin ne ürettiğini göstermeli — aksi halde neden
    # başarısız olduğu backend logundan hiç anlaşılmıyordu.
    with pytest.raises(ValueError, match="Modelin ürettiği metin"):
        _parse_quiz_json("Üzgünüm, bu dosyadan soru üretemiyorum.")


def test_quiz_max_tokens_scales_with_question_count_below_ceiling():
    # Az soru istendiğinde sabit bir üst sınır yerine soru sayısıyla ölçeklenen,
    # daha gerçekçi bir bütçe kullanılmalı — üretimi kısaltıyor.
    #
    # NOT (2026-08-17): bu test sabit `1000` bekliyordu ve KIRMIZIYDI.
    # `_QUIZ_MAX_TOKENS_CEILING` bir noktada 1000 -> 1200 yükseltilmiş,
    # `test_quiz_max_tokens_never_exceeds_ceiling` güncellenmiş ama bu test
    # güncellenmemişti. Sihirli sayı yerine artık sabitin kendisine
    # bakıyoruz ki aynı kayma tekrar yaşanmasın.
    assert _quiz_max_tokens(3) < _QUIZ_MAX_TOKENS_CEILING
    assert _quiz_max_tokens(1) < _quiz_max_tokens(3) < _quiz_max_tokens(5)


def test_quiz_max_tokens_never_exceeds_ceiling():
    # Çok sayıda soru istense bile (ör. arayüzdeki üst sınır: 15), 8GB
    # VRAM'li kartlardaki OOM güvenlik payını korumak için tavan aşılmıyor.
    assert _quiz_max_tokens(15) == _QUIZ_MAX_TOKENS_CEILING
    assert _quiz_max_tokens(1000) == _QUIZ_MAX_TOKENS_CEILING


def test_default_quiz_size_already_saturates_the_ceiling():
    """VARSAYILAN (5 soru) durumda bütçe tavana OTURUYOR — belgelenmiş bulgu.

    `_QUIZ_BASE_TOKENS + 5 * _QUIZ_TOKENS_PER_QUESTION = 100 + 1100 = 1200`,
    yani tam olarak tavan. Sonuç: "soru sayısına göre ölçekle" optimizasyonu
    varsayılan senaryoda hiçbir tasarruf SAĞLAMIYOR — 5 ve üzeri her istek
    aynı 1200 token bütçesini alıyor.

    Bu test bir DAVRANIŞ SÖZLEŞMESİ değil, bir GÖZLEM kaydı: Faz 3'te (LLM
    latency) bu sayılar ayarlanırken durumun bilinçli olarak değiştiğini
    görebilmek için duruyor. Kırılırsa bu bir hata değil — bütçeler kasıtlı
    değiştirilmiş demektir, o zaman bu test de güncellenmeli.
    """
    assert _quiz_max_tokens(5) == _QUIZ_MAX_TOKENS_CEILING


def test_parse_quiz_json_normalizes_null_options_and_answer_object():
    raw = '[{"question": "S?", "options": ["A", null, "C"], "answer": {"giriş": "C", "açıklama": "Doğru."}}]'
    assert _parse_quiz_json(raw) == [{
        "question": "S?",
        "options": ["A", "C"],
        "answer": "C Doğru.",
    }]
