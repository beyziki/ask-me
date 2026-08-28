"""Tüm test oturumu için ortak kurulum ve — en önemlisi — VERİTABANI İZOLASYONU.

NEDEN BU DOSYA VAR
------------------
Bu dosya eklenmeden önce testler ÜRETİM veritabanına yazıyordu. Kanıt:
`data/ask_me.db` içindeki 9 kullanıcıdan 6'sı test fixture'ıydı
(`alice_docs`, `alice_ml`, `bob_docs`, `carol_en`, `carol_tr`, `dup_user`) —
gerçek kullanıcılar yalnızca `byzerdem`, `beyza`, `ben`.

Mekanizma şuydu: `backend.app.core.config.settings` MODÜL IMPORT ZAMANINDA
bir kez oluşturuluyor. Bazı test dosyaları (`test_multiuser.py`,
`test_document_groups.py`, `test_rag_index.py`, `test_summary_utils.py`,
`test_ask_retrieval.py`) import etmeden önce `DATABASE_URL` env değişkenini
ayarlıyordu — ama bazıları (`test_ingestion.py`, `test_llm_utils.py`,
`test_quiz_utils.py`, `test_rag_merge.py`) ayarlamıyordu. pytest modülleri
alfabetik topladığı için, ayarlamayan bir modül önce yüklendiğinde `settings`
gerçek `data/ask_me.db`'ye bağlanıyor ve SONRAKİ tüm modüllerin env yazması
etkisiz kalıyordu (nesne zaten kurulmuş).

Sonuç sadece kirlilik değildi: `test_document_groups.py` içindeki silme testi
üretim veritabanından DOKÜMAN SİLİYORDU.

ÇÖZÜM
-----
pytest, `conftest.py`'yi HER test modülünden ÖNCE yükler. Bu yüzden env
değişkenlerini burada, modül seviyesinde (import zamanında) ayarlamak
`settings`'in her koşulda geçici bir dizine bağlanmasını garanti ediyor —
hangi test dosyasının önce yüklendiğinden bağımsız olarak.

`os.environ.setdefault` DEĞİL, doğrudan atama kullanılıyor: dışarıdan sızmış
bir `DATABASE_URL` (ör. geliştiricinin kabuğunda ihraç ettiği) izolasyonu
bozmasın.

Ek olarak `_fail_fast_if_not_isolated` fixture'ı her oturumda bunu bir kez
DOĞRULUYOR: gelecekte biri bu düzeni bozarsa testler sessizce üretime yazmak
yerine anında ve açık bir mesajla kırmızıya döner.
"""
from __future__ import annotations

import functools
import inspect
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# --- 1) İzolasyon: HER ŞEYDEN ÖNCE ---------------------------------------
# Bu blok, herhangi bir test modülü (ve dolayısıyla `backend.app.core.config`)
# import edilmeden önce çalışmak ZORUNDA. pytest'in conftest yükleme sırası
# bunu garanti ediyor.

_SESSION_DATA_DIR = Path(tempfile.mkdtemp(prefix="ask-me-tests-"))

os.environ["DATA_DIR"] = str(_SESSION_DATA_DIR)
os.environ["UPLOADS_DIR"] = str(_SESSION_DATA_DIR / "uploads")
os.environ["VECTORSTORE_DIR"] = str(_SESSION_DATA_DIR / "vectorstore")
os.environ["DATABASE_URL"] = f"sqlite:///{_SESSION_DATA_DIR / 'test.db'}"
# Testlerde Foundry Local / embedding ön ısıtması istemiyoruz (bkz. config.py).
os.environ["WARMUP_ON_STARTUP"] = "false"

# Proje kökünü sys.path'e ekle. Mevcut test dosyaları bunu kendileri de
# yapıyor (kalsın, zararsız); burada yapmak `pytest tests/tek_dosya.py`
# çağrısının da çalışmasını sağlıyor.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# --- 2) İzolasyonun gerçekten tuttuğunu doğrula ---------------------------


@pytest.fixture(scope="session", autouse=True)
def _fail_fast_if_not_isolated():
    """Testlerin üretim veritabanına bağlanmadığını oturum başında doğrular.

    Bu bilinçli olarak bir "güvenlik ağı" değil, bir ALARM: buraya düşülüyorsa
    yukarıdaki import-zamanı kurulumu bozulmuş demektir ve testler veri
    kaybettirebilir.
    """
    from backend.app.core.config import settings

    db_url = str(settings.database_url)
    assert str(_SESSION_DATA_DIR) in db_url or "ask-me-test" in db_url, (
        "TEST İZOLASYONU BOZUK: testler üretim veritabanına bağlı!\n"
        f"  settings.database_url = {db_url}\n"
        f"  beklenen dizin        = {_SESSION_DATA_DIR}\n"
        "Bir test modülü `backend.app.core.config`'i conftest.py'den önce "
        "import etmiş olabilir. Testleri ÇALIŞTIRMAYIN, önce bunu düzeltin."
    )
    assert "ask_me.db" not in db_url, (
        f"TEST İZOLASYONU BOZUK: üretim veritabanı kullanılıyor ({db_url})"
    )
    yield
    shutil.rmtree(_SESSION_DATA_DIR, ignore_errors=True)


# --- 3) Embedding modeli yoksa, ona bağlı testleri ATLA -------------------
# `/documents/upload` gerçek embedding üretiyor (bkz. rag.py:add_chunks_to_index).
# Model yerelde yoksa sentence-transformers onu HuggingFace'ten indirmeye
# çalışıyor; ağı kapalı bir CI runner'ında (ya da kurumsal bir proxy arkasında)
# bu 403/timeout ile patlıyor ve testler HATA veriyor — oysa doğru davranış
# ATLAMAK: bu bir kod hatası değil, ortam eksikliği.
#
# Hangi testlerin etkilendiğini elle listelemek yerine test fonksiyonunun
# KAYNAK KODUNA bakıyoruz: yükleme yapan her test er ya da geç
# `/documents/upload` çağırıyor. Böylece ileride eklenen yükleme testleri de
# otomatik kapsanıyor, listeyi güncellemeyi unutma riski yok.

_UPLOAD_MARKERS = ("/documents/upload", "_upload(")


@functools.lru_cache(maxsize=1)
def _embedding_model_available() -> bool:
    try:
        from backend.app.services.rag import get_embedder

        get_embedder()
        return True
    except Exception:
        return False


def _test_needs_embedding(item: pytest.Item) -> bool:
    func = getattr(item, "function", None)
    if func is None:
        return False
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return False
    if any(marker in source for marker in _UPLOAD_MARKERS):
        return True
    # Testin çağırdığı modül-seviyesi yardımcıları da tara (ör. `_upload`).
    module_source = ""
    try:
        module_source = inspect.getsource(inspect.getmodule(func))
    except (OSError, TypeError):
        return False
    for helper in ("_upload", "_upload_document", "upload_doc"):
        if f"{helper}(" in source and f"def {helper}(" in module_source:
            return True
    return False


def pytest_collection_modifyitems(config, items):
    """Embedding modeli yoksa, ona ihtiyaç duyan testleri atla."""
    candidates = [item for item in items if _test_needs_embedding(item)]
    if not candidates:
        return
    if _embedding_model_available():
        # Model var: bu testler gerçekten yavaş (her biri gerçek embedding
        # üretiyor). `slow` işaretle ki hızlı geliştirme döngüsü bunları
        # varsayılan olarak atlasın; `pytest -m ""` ile hepsi çalışır.
        for item in candidates:
            item.add_marker(pytest.mark.slow)
        return
    skip = pytest.mark.skip(
        reason=(
            "Embedding modeli yüklenemedi (yerel önbellekte yok ve indirilemiyor). "
            "Bu testler gerçek embedding üretimi gerektiriyor; backend'i en az bir "
            "kez çalıştırıp modeli önbelleğe alın."
        )
    )
    for item in candidates:
        item.add_marker(skip)


# --- 4) Ortak fixture'lar -------------------------------------------------


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Tek bir teste ait, izole bir veri dizini (uploads/ + vectorstore/)."""
    (tmp_path / "uploads").mkdir(exist_ok=True)
    (tmp_path / "vectorstore").mkdir(exist_ok=True)
    return tmp_path


@pytest.fixture
def clean_rag_caches():
    """`rag.py`'deki süreç-içi önbellekleri (FAISS index + BM25) temizler.

    Bu önbellekler kasıtlı olarak süreç ömrü boyunca yaşıyor (bkz.
    rag.py:_index_cache, _bm25_cache). Test içinde index dosyasını elle
    değiştiren senaryolarda, önceki testten kalan bir girdi yanlış sonuç
    verebilir; bu fixture o riski ortadan kaldırıyor.
    """
    from backend.app.services import rag

    def _clear():
        with rag._index_cache_lock:
            rag._index_cache.clear()
        with rag._bm25_cache_lock:
            rag._bm25_cache.clear()

    _clear()
    yield
    _clear()


@pytest.fixture(scope="session")
def production_db_path() -> Path:
    """Gerçek `data/ask_me.db`'nin yolu — YALNIZCA OKUMA amaçlı.

    Benchmark/eval testleri gerçek korpus üzerinde ölçüm yapmak isteyebilir.
    Bu fixture'ı kullanan hiçbir test veritabanına YAZMAMALI; yazan bir test
    varsa kendi kopyasını almalı.
    """
    path = _PROJECT_ROOT / "data" / "ask_me.db"
    if not path.exists():
        pytest.skip("Üretim veritabanı yok (data/ask_me.db)")
    return path
