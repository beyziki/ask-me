"""Çoklu kullanıcı izolasyonu ve TR/EN dil tercihi için entegrasyon testi.

Adım 7 (Çoklu kullanıcı ve TR/EN dil desteği) için doğrulama: gerçek FastAPI
uygulamasını (backend.app.main.app) TestClient ile çalıştırıp iki farklı
kullanıcının birbirinin dokümanlarını göremediğini ve dil tercihinin
kullanıcı bazlı saklandığını test ediyoruz.

Not: Bu test kasıtlı olarak Foundry Local'e ihtiyaç duymuyor — /ask, /quiz,
/code/explain uçlarındaki LLM çağrılarını değil, bunlardan ÖNCE çalışan
kullanıcı/doküman izolasyon mantığını test ediyor (ör. /quiz uç noktasında
sahiplik kontrolü LLM'e hiç gitmeden 404 döndürüyor). Bu sayede Foundry
Local kurulu olmayan bir ortamda (ör. CI) da çalışabiliyor.

Üretim veritabanı/vectorstore'a dokunmamak için data_dir'i geçici bir klasöre
yönlendiriyoruz. Bu env değişkenleri, `backend.app.core.config` (dolayısıyla
`backend.app.main`) import edilmeden ÖNCE ayarlanmalı; çünkü `settings`
modül import zamanında bir kere oluşturuluyor.
"""
import os
import tempfile
from pathlib import Path

_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="ask-me-test-"))
os.environ["DATA_DIR"] = str(_TEST_DATA_DIR)
os.environ["UPLOADS_DIR"] = str(_TEST_DATA_DIR / "uploads")
os.environ["VECTORSTORE_DIR"] = str(_TEST_DATA_DIR / "vectorstore")
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DATA_DIR / 'test.db'}"

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402

client = TestClient(app)


def _register(username: str, password: str, language: str) -> dict:
    resp = client.post(
        "/users/register",
        json={"username": username, "password": password, "preferred_language": language},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_register_and_login_roundtrip():
    user = _register("alice_ml", "s3cret-pw", "tr")
    assert user["username"] == "alice_ml"
    assert user["preferred_language"] == "tr"

    ok = client.post("/users/login", json={"username": "alice_ml", "password": "s3cret-pw"})
    assert ok.status_code == 200
    assert ok.json()["id"] == user["id"]

    wrong = client.post("/users/login", json={"username": "alice_ml", "password": "yanlis-sifre"})
    assert wrong.status_code == 401


def test_duplicate_username_is_rejected():
    _register("dup_user", "s3cret-pw", "tr")
    resp = client.post(
        "/users/register",
        json={"username": "dup_user", "password": "baska-sifre", "preferred_language": "en"},
    )
    assert resp.status_code == 400


def test_documents_are_isolated_between_users():
    # Bu test dosya yüklerken embedding modelini (sentence-transformers,
    # bkz. core/config.py:embedding_model) devreye sokar. İlk çalıştırmada
    # model HuggingFace Hub'dan indirilir ve yerelde önbelleğe alınır
    # (backend'i daha önce en az bir kez çalıştırdıysan zaten önbellekte
    # olmalı, tıpkı uvicorn loglarındaki "Loading weights..." adımı gibi).
    alice = _register("alice_docs", "s3cret-pw", "tr")
    bob = _register("bob_docs", "s3cret-pw", "en")

    upload = client.post(
        "/documents/upload",
        headers={"X-User-Id": str(alice["id"])},
        files={
            "file": (
                "notes.md",
                b"# Turing Machines\nBir Turing makinesi sonsuz bir seride calisir.",
                "text/markdown",
            )
        },
    )
    assert upload.status_code == 200, upload.text
    doc_id = upload.json()["id"]

    # Alice kendi dokumanini gorebilmeli
    alice_docs = client.get("/documents", headers={"X-User-Id": str(alice["id"])})
    assert alice_docs.status_code == 200
    assert [d["id"] for d in alice_docs.json()] == [doc_id]

    # Bob'un listesi bos olmali (izolasyon)
    bob_docs = client.get("/documents", headers={"X-User-Id": str(bob["id"])})
    assert bob_docs.status_code == 200
    assert bob_docs.json() == []

    # Bob, Alice'in dokumanina ID'sini bilse bile erisemiyor: sahiplik
    # kontrolu LLM cagrisindan (generate_quiz) ONCE calisip 404 donuyor.
    quiz_attempt = client.post(
        "/quiz",
        headers={"X-User-Id": str(bob["id"])},
        json={"document_id": doc_id, "num_questions": 3},
    )
    assert quiz_attempt.status_code == 404


def test_language_preference_is_per_user():
    tr_user = _register("carol_tr", "s3cret-pw", "tr")
    en_user = _register("carol_en", "s3cret-pw", "en")
    assert tr_user["preferred_language"] == "tr"
    assert en_user["preferred_language"] == "en"
