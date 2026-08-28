"""Hybrid RAG alma (retrieval) adımı için testler: `_dedupe_by_content`.

Gözlemlenen bir kullanıcı ekran görüntüsü, aynı dosyanın (ör. yanlışlıkla)
birden fazla kez yüklenmesi durumunda "Kaynaklar" listesinde İÇERİĞİ
birebir aynı parçaların art arda tekrarlandığını gösterdi. `hybrid_merge`
zaten `chunk_id` bazında dedupe ediyor, ama bu tür kopyalar farklı
`chunk_id`'lere sahip olduğu için orada yakalanmıyor -- bu yüzden
`backend/app/api/ask.py:_dedupe_by_content` eklendi, burada da test
ediliyor.

Not: `_dedupe_by_content` saf bir fonksiyon (DB/Foundry Local gerektirmiyor),
bu yüzden test_llm_utils.py'deki gibi doğrudan birim testle doğrulanabiliyor.
`_retrieve`'in kendisi (LLM'e hiç gitmeden, yalnızca hybrid RAG + DB
kullanan kısmı) ise test_document_groups.py/test_multiuser.py'deki gibi
gerçek FastAPI uygulaması üzerinden test ediliyor.

ÖNEMLİ: `backend.app.api.ask` içe aktarımı, transitif olarak
`backend.app.db.base`'i (ve oradaki modül seviyesindeki `create_engine`
çağrısını) tetikliyor -- bu da import ANINDAKİ ortam değişkenlerine göre
KALICI bir engine/DB'ye bağlanıyor (pytest tüm test dosyalarını AYNI
süreçte topladığı için bu engine tüm test oturumu boyunca paylaşılıyor).
Bu yüzden test_document_groups.py/test_multiuser.py'deki gibi, HERHANGİ
bir `backend.app` import'undan ÖNCE izole bir geçici DB'ye yönlendiriyoruz
-- aksi halde (gözlemlendi) bu dosya alfabetik sırada diğerlerinden önce
toplandığı için varsayılan/üretim DB'sine bağlanıp tüm test oturumunu
bozabiliyordu.
"""
import os
import tempfile
from pathlib import Path

_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="ask-me-test-ask-retrieval-"))
os.environ.setdefault("DATA_DIR", str(_TEST_DATA_DIR))
os.environ.setdefault("UPLOADS_DIR", str(_TEST_DATA_DIR / "uploads"))
os.environ.setdefault("VECTORSTORE_DIR", str(_TEST_DATA_DIR / "vectorstore"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TEST_DATA_DIR / 'test.db'}")

from backend.app.api.ask import (  # noqa: E402
    _cap_context,
    _dedupe_by_content,
    _detect_language,
)
from backend.app.services.rag import SearchHit  # noqa: E402


def test_dedupe_by_content_drops_hits_with_identical_text():
    hits = [
        SearchHit(chunk_id=1, score=0.9),
        SearchHit(chunk_id=2, score=0.8),  # chunk_id=1 ile aynı metin -> düşmeli
        SearchHit(chunk_id=3, score=0.7),
    ]
    chunk_id_to_text = {
        1: "AUTOMATA THEORY AND FORMAL LANGUAGES ... Turing Machines Part2",
        2: "AUTOMATA THEORY AND FORMAL LANGUAGES ... Turing Machines Part2",
        3: "Farklı bir parça: nondeterministic Turing machine has an equivalent...",
    }
    deduped = _dedupe_by_content(hits, chunk_id_to_text)
    assert [h.chunk_id for h in deduped] == [1, 3]


def test_dedupe_by_content_keeps_highest_scored_occurrence_first():
    # Sıra zaten skora göre azalan geldiği için (bkz. hybrid_merge), dedupe
    # her zaman İLK (en yüksek skorlu) tekrarı tutmalı.
    hits = [SearchHit(chunk_id=10, score=0.95), SearchHit(chunk_id=11, score=0.5)]
    chunk_id_to_text = {10: "aynı metin", 11: "aynı metin"}
    deduped = _dedupe_by_content(hits, chunk_id_to_text)
    assert len(deduped) == 1
    assert deduped[0].chunk_id == 10


def test_dedupe_by_content_skips_hits_missing_from_text_map():
    # `_retrieve`'de bir chunk silinmiş/erişilemez olabilir; bu durumda
    # `chunk_id_to_text.get` None döner ve hit sessizce atlanmalı (crash yok).
    hits = [SearchHit(chunk_id=1, score=0.9), SearchHit(chunk_id=2, score=0.8)]
    chunk_id_to_text = {1: "metin"}
    deduped = _dedupe_by_content(hits, chunk_id_to_text)
    assert [h.chunk_id for h in deduped] == [1]


def test_dedupe_by_content_no_op_when_all_texts_distinct():
    hits = [SearchHit(chunk_id=1, score=0.9), SearchHit(chunk_id=2, score=0.8)]
    chunk_id_to_text = {1: "birinci", 2: "ikinci"}
    deduped = _dedupe_by_content(hits, chunk_id_to_text)
    assert [h.chunk_id for h in deduped] == [1, 2]


# --- Context boyutu sınırı ------------------------------------------------
# Prefill (modelin context'i okuma) süresi context uzunluğuyla doğru
# orantılı; sınırsız büyüyen bir context ilk token'ın ekrana gelmesini
# gözle görülür biçimde geciktiriyordu.


def test_cap_context_keeps_chunks_until_char_limit():
    chunks = ["a" * 100, "b" * 100, "c" * 100]
    assert _cap_context(chunks, max_chars=250) == [chunks[0], chunks[1]]


def test_cap_context_always_keeps_at_least_one_chunk():
    # Tek bir parça bile sınırdan büyükse yine de gönderiliyor: aksi halde
    # context tamamen boşalır ve cevap hiç kaynağa dayanmaz.
    chunks = ["x" * 500]
    assert _cap_context(chunks, max_chars=100) == chunks


def test_cap_context_no_op_when_under_limit():
    chunks = ["kısa", "parçalar"]
    assert _cap_context(chunks, max_chars=1000) == chunks


# --- Dil tespiti ----------------------------------------------------------
# `langdetect` kısa metinlerde tutarsız; Türkçe bir soruya İngilizce cevap
# dönmesine yol açıyordu (gözlemlenen bir kalite şikâyeti).


def test_detect_language_returns_tr_for_turkish_specific_letters():
    assert _detect_language("Yığın ve kuyruk arasındaki fark nedir?") == "tr"


def test_detect_language_defaults_to_tr_for_short_ambiguous_text():
    # Türkçeye özgü harf içermeyen ama çok kısa olan sorularda istatistiksel
    # tahmine güvenmiyoruz; uygulamanın birincil dili olan Türkçe kalıyor.
    assert _detect_language("stack nedir") == "tr"
