"""Yüklenen notlardan otomatik quiz / çalışma sorusu üretimi.

LLM'e (Foundry Local) yapılandırılmış bir prompt vererek JSON formatında
soru-cevap listesi üretilir.
"""
from __future__ import annotations

import json
import re

from backend.app.services.llm import (
    FoundryNotAvailable,
    TRANSPORT_ERRORS,
    SYSTEM_PROMPT_QUIZ_EN,
    SYSTEM_PROMPT_QUIZ_TR,
    _create_chat_completion,
    _get_manager,
    _get_model_id,
    _stream_and_strip,
    build_prompt,
    strip_think,
)

QUIZ_PROMPT_TR = (
    "Aşağıdaki kaynak metinlerden yararlanarak {n} adet çalışma sorusu üret. "
    "Yalnızca geçerli JSON döndür, başka hiçbir açıklama ekleme. Her soru kısa "
    "olsun ve answer TEK bir string olsun; answer nesnesi veya ek alan kullanma. "
    "Tam olarak şu formatı kullan:\n"
    '[{{"question": "...", "options": ["...", "...", "...", "..."] veya null, "answer": "..."}}]'
)

QUIZ_PROMPT_EN = (
    "Using the source excerpts below, generate {n} study questions. "
    "Return ONLY valid JSON, no other text. Keep each question short and make "
    "answer a SINGLE string; do not use an answer object or extra fields. Format:\n"
    '[{{"question": "...", "options": ["...", "...", "...", "..."] or null, "answer": "..."}}]'
)

# Soru başına ayrılan kaba token bütçesi (soru metni + ~4 seçenek + cevap +
# JSON noktalama işaretleri) ve sabit bir taban (sistem/JSON iskeleti için).
# Az soru istendiğinde (varsayılan: 5) modele hep aynı geniş üst sınırı
# (eskiden sabit 1000) vermek üretimi gereksiz uzatıyordu; bunun yerine
# istenen soru sayısına göre ölçekleniyoruz. Üst sınır (1000) ise 8GB VRAM'li
# kartlardaki OOM güvenlik payını korumak için AŞILMIYOR — bkz.
# backend/app/api/quiz.py:MAX_QUIZ_CHUNKS'taki OOM notu; bu sınır zaten bir
# kez 1400'den 1000'e düşürülmüştü.
_QUIZ_TOKENS_PER_QUESTION = 220
_QUIZ_BASE_TOKENS = 100
_QUIZ_MAX_TOKENS_CEILING = 1200


def _quiz_max_tokens(num_questions: int) -> int:
    return min(_QUIZ_MAX_TOKENS_CEILING, _QUIZ_BASE_TOKENS + num_questions * _QUIZ_TOKENS_PER_QUESTION)


# Modelin JSON'ı bir kod bloğuna sarması sık görülen bir davranış (prompt'ta
# "kod bloğu ekleme" dense bile).
_FENCE_OPEN_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*")
_FENCE_CLOSE_RE = re.compile(r"\s*```\s*$")


def _strip_code_fence(text: str) -> str:
    return _FENCE_CLOSE_RE.sub("", _FENCE_OPEN_RE.sub("", text)).strip()


def _extract_json_payload(text: str) -> str | None:
    """Metnin İÇİNDEN ilk dengeli JSON dizisini/nesnesini çıkarır.

    Gözlemlenen arıza: model JSON'dan önce ya da sonra düz metin ekliyor
    ("İşte sorular:", "Umarım yardımcı olur." gibi). Bu durumda `json.loads`
    "Expecting value: line 1 column 1 (char 0)" hatası veriyor ve quiz
    tamamen kayboluyordu -- oysa asıl JSON metnin içinde sapasağlam duruyor.

    Basit bir `find('[')` + `rfind(']')` yeterli DEĞİL: soru metinlerinin
    İÇİNDE de köşeli parantez geçebiliyor (ör. "[Kaynak 2]"). Bu yüzden
    parantezleri gerçekten sayıyoruz ve string literal'lerin (ve kaçış
    karakterlerinin) içindeki parantezleri saymıyoruz.
    """
    start = None
    for i, ch in enumerate(text):
        if ch in "[{":
            start = i
            break
    if start is None:
        return None

    opener = text[start]
    closer = "]" if opener == "[" else "}"
    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _coerce_question_list(data) -> list[dict]:
    """Ayrıştırılmış JSON'ı soru listesine indirger.

    Model bazen istenen düz diziyi değil, onu saran bir nesne döndürüyor:
    `{"questions": [...]}` gibi. İçerik doğru olduğu hâlde bunu reddetmek
    gereksiz bir başarısızlık olur, bu yüzden kabul ediyoruz.
    """
    if isinstance(data, list):
        questions = data
    if isinstance(data, dict):
        for key in ("questions", "quiz", "sorular", "items", "data"):
            value = data.get(key)
            if isinstance(value, list):
                questions = value
                break
        # Tek bir soru nesnesi döndürüldüyse tek elemanlı listeye sar.
        if "questions" not in locals() and {"question", "answer"} <= set(data):
            questions = [data]
    if "questions" not in locals():
        questions = None
    if not isinstance(questions, list):
        raise ValueError(f"Beklenen soru listesi yerine {type(data).__name__} geldi")
    normalized = []
    for question in questions:
        if not isinstance(question, dict) or "question" not in question or "answer" not in question:
            raise ValueError("Soru question ve answer alanlarını içermiyor")
        options = question.get("options")
        if isinstance(options, list):
            options = [str(option) for option in options if option is not None] or None
        elif options is not None:
            options = [str(options)]
        answer = question["answer"]
        if isinstance(answer, dict):
            answer = " ".join(str(value) for value in answer.values() if value is not None)
        normalized.append({**question, "options": options, "answer": str(answer)})
    return normalized


def _parse_quiz_json(raw: str) -> list[dict]:
    """Modelin ürettiği (zaten `<think>`'ten temizlenmiş) ham metni quiz
    JSON'ına çevirir.

    `generate_quiz` (tek seferde) ve `/quiz/stream` (streaming -- tüm parçalar
    biriktirildikten SONRA) tarafından ortak kullanılıyor.

    Yerel modeller "SADECE JSON döndür" talimatına her zaman uymuyor; bu
    yüzden kademeli deniyoruz: (1) doğrudan, (2) kod bloğu işaretlerini
    temizleyip, (3) metnin içinden dengeli JSON parçasını çıkarıp. Hepsi
    başarısız olursa, teşhisi kolaylaştırmak için ham metnin başını da içeren
    açık bir hata veriyoruz (eski davranışta yalnızca "Expecting value: line 1
    column 1" görünüyordu, bu da modelin ne ürettiğini hiç göstermiyordu).
    """
    if not raw or not raw.strip():
        raise ValueError("Model hiç çıktı üretmedi (boş yanıt)")

    candidates = [raw, _strip_code_fence(raw)]
    payload = _extract_json_payload(raw)
    if payload:
        candidates.append(payload)

    for candidate in candidates:
        if not candidate or not candidate.strip():
            continue
        try:
            return _coerce_question_list(json.loads(candidate))
        except (json.JSONDecodeError, ValueError):
            continue

    snippet = raw.strip()[:200]
    raise ValueError(f"Geçerli JSON bulunamadı. Modelin ürettiği metin: {snippet!r}")


def generate_quiz(context_chunks: list[str], num_questions: int = 5, language: str = "tr") -> list[dict]:
    """Tek seferde (streaming olmayan) quiz üretimi. Programatik kullanım
    için korunuyor; asıl arayüz artık `generate_quiz_stream`'i kullanıyor."""
    manager = _get_manager()
    model_id = _get_model_id()
    instruction = (QUIZ_PROMPT_TR if language == "tr" else QUIZ_PROMPT_EN).format(n=num_questions)
    # Quiz'e özel sistem prompt'u: normal cevap prompt'undaki "[Kaynak N]
    # biçiminde atıf yap" talimatı, JSON olarak parse edilen quiz çıktısını
    # bozabiliyor (bkz. llm.py:SYSTEM_PROMPT_QUIZ_TR).
    messages = build_prompt(
        instruction,
        context_chunks,
        language,
        system_prompt=SYSTEM_PROMPT_QUIZ_TR if language == "tr" else SYSTEM_PROMPT_QUIZ_EN,
    )

    from openai import OpenAI

    client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key)
    # `_create_chat_completion` üzerinden geçiyoruz ki thinking modelinde
    # `enable_thinking=False` denemesi burada da yapılsın, düşünmeyen modelde
    # ise hiç gönderilmesin (bkz. llm.py).
    response = _create_chat_completion(
        client,
        model=model_id,
        messages=messages,
        temperature=0.4,
        max_tokens=_quiz_max_tokens(num_questions),
    )
    # /no_think switch'ine rağmen model yine de <think> bloğu üretebilir;
    # JSON parse etmeden önce temizliyoruz (aksi halde json.loads kırılır).
    raw = strip_think(response.choices[0].message.content)
    return _parse_quiz_json(raw)


def generate_quiz_stream(context_chunks: list[str], num_questions: int = 5, language: str = "tr"):
    """`generate_quiz`'in streaming (token token) karşılığı.

    Model çıktısı ham JSON metni olduğu ve parça parça geçerli bir JSON
    render edilemeyeceği için burada JSON PARSE ETMİYORUZ; yalnızca
    `<think>` bloğu temizlenmiş ham metin parçalarını yield ediyoruz —
    frontend bunları bir ilerleme göstergesi (ör. "N karakter alındı") için
    kullanıyor. Çağıran taraf (`backend/app/api/quiz.py:create_quiz_stream`),
    tüm parçaları birleştirip akış bittikten SONRA `_parse_quiz_json` ile
    ayrıştırır. Bağlantı üretim sırasında/sonunda beklenmedik şekilde koparsa
    (bkz. llm.py:_stream_and_strip) ve hiç içerik gelmediyse
    `FoundryNotAvailable` fırlatır.
    """
    manager = _get_manager()
    model_id = _get_model_id()
    instruction = (QUIZ_PROMPT_TR if language == "tr" else QUIZ_PROMPT_EN).format(n=num_questions)
    messages = build_prompt(
        instruction,
        context_chunks,
        language,
        system_prompt=SYSTEM_PROMPT_QUIZ_TR if language == "tr" else SYSTEM_PROMPT_QUIZ_EN,
    )

    from openai import OpenAI

    client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key)
    stream = _create_chat_completion(
        client,
        model=model_id,
        messages=messages,
        temperature=0.4,
        max_tokens=_quiz_max_tokens(num_questions),
        stream=True,
    )

    try:
        yield from _stream_and_strip(stream)
    except TRANSPORT_ERRORS as exc:
        raise FoundryNotAvailable(
            "Foundry Local ile bağlantı üretim başlamadan/sırasında beklenmedik şekilde "
            "kesildi. `foundry server status` ile servisi kontrol edip tekrar deneyin."
        ) from exc
