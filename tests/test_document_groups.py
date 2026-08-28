"""Doküman gruplama (klasörleme) uçları için entegrasyon testi.

test_multiuser.py'deki gibi gerçek FastAPI uygulamasını (backend.app.main.app)
TestClient ile, geçici bir SQLite/vectorstore ile çalıştırıyoruz. Bu testler
de (test_multiuser.py'deki gibi) kasıtlı olarak Foundry Local'e ihtiyaç
duymuyor; grup CRUD'u ve doküman-grup ilişkisi LLM'e hiç gitmeden test
edilebiliyor.
"""
import os
import tempfile
from pathlib import Path

_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="ask-me-test-groups-"))
os.environ.setdefault("DATA_DIR", str(_TEST_DATA_DIR))
os.environ.setdefault("UPLOADS_DIR", str(_TEST_DATA_DIR / "uploads"))
os.environ.setdefault("VECTORSTORE_DIR", str(_TEST_DATA_DIR / "vectorstore"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TEST_DATA_DIR / 'test.db'}")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402

client = TestClient(app)


def _register(username: str) -> dict:
    resp = client.post(
        "/users/register",
        json={"username": username, "password": "s3cret-pw", "preferred_language": "tr"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _upload(user_id: int, filename: str = "notes.md", group_id: int | None = None) -> dict:
    data = {} if group_id is None else {"group_id": str(group_id)}
    resp = client.post(
        "/documents/upload",
        headers={"X-User-Id": str(user_id)},
        files={"file": (filename, b"# Baslik\nBirkac satir icerik.", "text/markdown")},
        data=data,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_create_and_list_document_group():
    user = _register("grp_owner")
    resp = client.post(
        "/documents/groups",
        headers={"X-User-Id": str(user["id"])},
        json={"name": "Bilgisayar Ağları"},
    )
    assert resp.status_code == 200, resp.text
    group = resp.json()
    assert group["name"] == "Bilgisayar Ağları"

    listed = client.get("/documents/groups", headers={"X-User-Id": str(user["id"])})
    assert listed.status_code == 200
    assert [g["id"] for g in listed.json()] == [group["id"]]


def test_group_name_cannot_be_blank():
    user = _register("grp_blank")
    resp = client.post(
        "/documents/groups",
        headers={"X-User-Id": str(user["id"])},
        json={"name": "   "},
    )
    assert resp.status_code == 422


def test_groups_are_isolated_between_users():
    alice = _register("grp_alice")
    bob = _register("grp_bob")

    resp = client.post(
        "/documents/groups",
        headers={"X-User-Id": str(alice["id"])},
        json={"name": "Alice'in grubu"},
    )
    group_id = resp.json()["id"]

    # Bob'un grup listesi boş olmalı (izolasyon).
    bob_groups = client.get("/documents/groups", headers={"X-User-Id": str(bob["id"])})
    assert bob_groups.json() == []

    # Bob, Alice'in grubunu ID'sini bilse bile silemez.
    delete_attempt = client.delete(
        f"/documents/groups/{group_id}", headers={"X-User-Id": str(bob["id"])}
    )
    assert delete_attempt.status_code == 404

    # Bob kendi dokümanını Alice'in grubuna atayamaz.
    bob_doc = _upload(bob["id"], "bob-notes.md")
    assign_attempt = client.patch(
        f"/documents/{bob_doc['id']}/group",
        headers={"X-User-Id": str(bob["id"])},
        json={"group_id": group_id},
    )
    assert assign_attempt.status_code == 404


def test_upload_with_group_assigns_immediately():
    user = _register("grp_upload")
    group = client.post(
        "/documents/groups",
        headers={"X-User-Id": str(user["id"])},
        json={"name": "Ders 1"},
    ).json()

    doc = _upload(user["id"], "ders1-notlari.md", group_id=group["id"])
    assert doc["group_id"] == group["id"]


def test_reassign_and_ungroup_document():
    user = _register("grp_reassign")
    group_a = client.post(
        "/documents/groups", headers={"X-User-Id": str(user["id"])}, json={"name": "A"}
    ).json()
    group_b = client.post(
        "/documents/groups", headers={"X-User-Id": str(user["id"])}, json={"name": "B"}
    ).json()
    doc = _upload(user["id"], "notes.md", group_id=group_a["id"])

    moved = client.patch(
        f"/documents/{doc['id']}/group",
        headers={"X-User-Id": str(user["id"])},
        json={"group_id": group_b["id"]},
    )
    assert moved.status_code == 200
    assert moved.json()["group_id"] == group_b["id"]

    ungrouped = client.patch(
        f"/documents/{doc['id']}/group",
        headers={"X-User-Id": str(user["id"])},
        json={"group_id": None},
    )
    assert ungrouped.status_code == 200
    assert ungrouped.json()["group_id"] is None


def test_deleting_group_ungroups_documents_without_deleting_them():
    user = _register("grp_delete")
    group = client.post(
        "/documents/groups", headers={"X-User-Id": str(user["id"])}, json={"name": "Geçici"}
    ).json()
    doc = _upload(user["id"], "notes.md", group_id=group["id"])

    delete_resp = client.delete(
        f"/documents/groups/{group['id']}", headers={"X-User-Id": str(user["id"])}
    )
    assert delete_resp.status_code == 204

    docs = client.get("/documents", headers={"X-User-Id": str(user["id"])}).json()
    assert [d["id"] for d in docs] == [doc["id"]]
    assert docs[0]["group_id"] is None


def test_delete_document_removes_it_and_ungroups_related_quiz():
    """DELETE /documents/{id}: sahibi olmayan 404 alır; sahibi sildiğinde
    doküman listeden kalkar, kalan dokümanlar etkilenmez, ve dokümana bağlı
    bir quiz varsa quiz SİLİNMİYOR yalnızca document_id'si null'a düşüyor
    (bkz. backend/app/api/documents.py:delete_document, rag.py:rebuild_index)."""
    user = _register("del_owner")
    other = _register("del_other")

    doc_a = _upload(user["id"], "a.md")
    doc_b = _upload(user["id"], "b.md")

    # Başka bir kullanıcı sahibi olmadığı dokümanı silemez.
    forbidden = client.delete(
        f"/documents/{doc_a['id']}", headers={"X-User-Id": str(other["id"])}
    )
    assert forbidden.status_code == 404

    # doc_a'ya bağlı bir quiz oluştur (LLM'e gitmeden, doğrudan DB üzerinden
    # değil — quiz endpoint'i Foundry Local gerektirdiği için burada
    # test_multiuser.py'deki gibi servis katmanını LLM'siz test etmek yerine,
    # quiz kaydını doğrudan modelle oluşturuyoruz).
    from backend.app.db.base import SessionLocal
    from backend.app.db.models import Quiz

    db = SessionLocal()
    try:
        quiz = Quiz(
            owner_id=user["id"],
            document_id=doc_a["id"],
            title="Test Quiz",
        )
        db.add(quiz)
        db.commit()
        db.refresh(quiz)
        quiz_id = quiz.id
    finally:
        db.close()

    delete_resp = client.delete(
        f"/documents/{doc_a['id']}", headers={"X-User-Id": str(user["id"])}
    )
    assert delete_resp.status_code == 204

    remaining = client.get("/documents", headers={"X-User-Id": str(user["id"])}).json()
    assert [d["id"] for d in remaining] == [doc_b["id"]]

    db = SessionLocal()
    try:
        refreshed_quiz = db.query(Quiz).get(quiz_id)
        assert refreshed_quiz is not None
        assert refreshed_quiz.document_id is None
    finally:
        db.close()

    # Var olmayan bir dokümanı silmeye çalışmak da 404 döner.
    missing = client.delete(
        f"/documents/{doc_a['id']}", headers={"X-User-Id": str(user["id"])}
    )
    assert missing.status_code == 404
