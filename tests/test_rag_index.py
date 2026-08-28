"""FAISS index bakımı için testler: bir doküman silindiğinde index'in
KALAN vektörler korunarak yeniden kurulması (bkz.
backend/app/services/rag.py:drop_rows_from_index).

Bu testler embedding modeline ihtiyaç duymuyor — vektörleri doğrudan
`add_chunks_to_index` yerine sahte bir embedder üzerinden yazıyoruz, böylece
ağ/model indirmesi olmadan da çalışıyorlar.
"""
import os
import tempfile
from pathlib import Path

_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="ask-me-test-rag-index-"))
os.environ.setdefault("DATA_DIR", str(_TEST_DATA_DIR))
os.environ.setdefault("UPLOADS_DIR", str(_TEST_DATA_DIR / "uploads"))
os.environ.setdefault("VECTORSTORE_DIR", str(_TEST_DATA_DIR / "vectorstore"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TEST_DATA_DIR / 'test.db'}")

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from backend.app.services import rag  # noqa: E402


def _write_index(user_id: int, vectors: np.ndarray) -> None:
    """Verilen vektörlerle kullanıcının index dosyasını sıfırdan yazar."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(rag._index_path(user_id)))


def _read_all(user_id: int) -> np.ndarray:
    index = faiss.read_index(str(rag._index_path(user_id)))
    return np.asarray(index.reconstruct_n(0, index.ntotal), dtype="float32")


@pytest.fixture
def user_id(tmp_path, monkeypatch):
    """Her test kendi vectorstore klasörünü kullansın (index dosyaları
    kullanıcı id'siyle adlandırıldığı için testler arası sızıntı olmasın)."""
    monkeypatch.setattr(rag.settings, "vectorstore_dir", tmp_path)
    rag._index_cache.clear()
    return 1


def _distinct_vectors(n: int, dim: int = 4) -> np.ndarray:
    """Her satırı birbirinden ayırt edilebilir vektörler (i. satır = i+1)."""
    return np.arange(1, n + 1, dtype="float32").reshape(n, 1) * np.ones((1, dim), dtype="float32")


def test_drop_rows_keeps_only_requested_rows_in_order(user_id):
    _write_index(user_id, _distinct_vectors(5))

    # 0. ve 3. satırlara sahip chunk'lar silindi; 1, 2, 4 kalıyor.
    new_rows = rag.drop_rows_from_index(user_id, [1, 2, 4])

    assert new_rows == [0, 1, 2]
    remaining = _read_all(user_id)
    assert remaining.shape[0] == 3
    # Kopyalanan vektörler DOĞRU satırlardan gelmeli: eski 1,2,4 -> 2,3,5
    assert [float(row[0]) for row in remaining] == [2.0, 3.0, 5.0]


def test_drop_rows_preserves_given_order(user_id):
    _write_index(user_id, _distinct_vectors(3))
    rag.drop_rows_from_index(user_id, [2, 0])
    assert [float(row[0]) for row in _read_all(user_id)] == [3.0, 1.0]


def test_drop_rows_deletes_index_file_when_nothing_remains(user_id):
    _write_index(user_id, _distinct_vectors(3))
    assert rag.drop_rows_from_index(user_id, []) == []
    # Bayat/yanlış boyutlu bir index dosyası kalıntı olarak kalmamalı.
    assert not rag._index_path(user_id).exists()


def test_drop_rows_returns_none_when_index_missing(user_id):
    # Index dosyası hiç yoksa hızlı yol uygulanamaz; çağıran taraf
    # `rebuild_index`'e düşsün diye None dönmeli.
    assert rag.drop_rows_from_index(user_id, [0, 1]) is None


def test_drop_rows_returns_none_on_out_of_range_row(user_id):
    # DB'deki vector_row'lar index ile tutarsızsa (ör. index elle silinip
    # yeniden kurulmuş) sessizce YANLIŞ vektör kopyalamak yerine hızlı
    # yoldan vazgeçmeli.
    _write_index(user_id, _distinct_vectors(2))
    assert rag.drop_rows_from_index(user_id, [0, 7]) is None


def test_drop_rows_returns_none_when_a_chunk_has_no_vector_row(user_id):
    _write_index(user_id, _distinct_vectors(2))
    assert rag.drop_rows_from_index(user_id, [0, None]) is None


def test_drop_rows_invalidates_cached_index(user_id):
    # `semantic_search` index'i önbellekten okuyor (bkz. _read_index_cached);
    # silme sonrası BAYAT index'i okumamalı.
    _write_index(user_id, _distinct_vectors(3))
    rag._read_index_cached(rag._index_path(user_id))  # önbelleğe al

    rag.drop_rows_from_index(user_id, [2])

    cached = rag._read_index_cached(rag._index_path(user_id))
    assert cached.ntotal == 1
    assert float(np.asarray(cached.reconstruct_n(0, 1))[0][0]) == 3.0


# --- Filtreli semantic arama ---------------------------------------------
# Regresyon: kullanıcı aramayı belirli dosyalarla sınırladığında (bkz.
# AskRequest.document_ids -> api/ask.py:_retrieve -> row_to_chunk_id),
# `semantic_search` FAISS'e GLOBAL bir arama yapıp kapsam dışı satırları
# sessizce atıyordu. Seçili dosyanın parçaları genel sıralamada ilk top_k
# içine giremezse sonuç BOŞ dönüyor, bu da has_context=False'a ve
# "yüklenen dosyalarda bir bölüm bulunamadı" cevabına yol açıyordu.


class _FakeEmbedder:
    """Sorguyu sabit bir vektöre çeviren sahte embedder (model indirmeden
    test edebilmek için)."""

    def __init__(self, vector):
        self._vector = np.asarray([vector], dtype="float32")

    def encode(self, texts, normalize_embeddings=True):
        return self._vector


def _unit_rows(n: int, dim: int = 4) -> np.ndarray:
    """Birbirine dik n vektör; i. satır yalnızca i. boyutta 1."""
    vectors = np.zeros((n, dim), dtype="float32")
    for i in range(n):
        vectors[i][i % dim] = 1.0
    return vectors


def test_semantic_search_only_scores_rows_in_scope(user_id, monkeypatch):
    # 4 satır, hepsi farklı boyutta 1. Sorgu 3. satıra (row=3) tam uyuyor.
    _write_index(user_id, _unit_rows(4))
    monkeypatch.setattr(rag, "_embedder", _FakeEmbedder([0.0, 0.0, 0.0, 1.0]))
    rag._index_cache.clear()

    # Kapsam YALNIZCA row 0 ve 1: en iyi eşleşme (row 3) kapsam dışı.
    hits = rag.semantic_search(user_id, "soru", top_k=2, row_to_chunk_id={0: 100, 1: 101})

    # ESKİ DAVRANIŞ: global top-2 (row 3 ve bir başkası) dönüp kapsam dışı
    # oldukları için elenirdi -> boş liste. Artık kapsam içinde aranıyor.
    assert [h.chunk_id for h in hits] == [100, 101]
    assert all(0.0 <= h.score <= 1.0 for h in hits)


def test_semantic_search_ranks_within_scope(user_id, monkeypatch):
    _write_index(user_id, _unit_rows(4))
    # Sorgu row 1'e row 0'dan daha yakın.
    monkeypatch.setattr(rag, "_embedder", _FakeEmbedder([0.3, 0.9, 0.0, 0.0]))
    rag._index_cache.clear()

    hits = rag.semantic_search(user_id, "soru", top_k=2, row_to_chunk_id={0: 100, 1: 101})

    assert [h.chunk_id for h in hits] == [101, 100]
    assert hits[0].score > hits[1].score


def test_semantic_search_full_scope_matches_faiss(user_id, monkeypatch):
    """Kapsam tüm index olduğunda hızlı yol (FAISS'in kendi araması)
    kullanılıyor; sonuç yine doğru sırada gelmeli."""
    _write_index(user_id, _unit_rows(4))
    monkeypatch.setattr(rag, "_embedder", _FakeEmbedder([0.0, 0.0, 1.0, 0.0]))
    rag._index_cache.clear()

    row_to_chunk_id = {0: 100, 1: 101, 2: 102, 3: 103}
    hits = rag.semantic_search(user_id, "soru", top_k=4, row_to_chunk_id=row_to_chunk_id)

    assert hits[0].chunk_id == 102
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_semantic_search_ignores_stale_rows(user_id, monkeypatch):
    """DB'deki vector_row'lar index'ten büyükse (bayat kayıt) patlamak
    yerine o satırları atlamalı."""
    _write_index(user_id, _unit_rows(2))
    monkeypatch.setattr(rag, "_embedder", _FakeEmbedder([1.0, 0.0, 0.0, 0.0]))
    rag._index_cache.clear()

    hits = rag.semantic_search(user_id, "soru", top_k=3, row_to_chunk_id={0: 100, 9: 999})

    assert [h.chunk_id for h in hits] == [100]


def test_semantic_search_returns_empty_for_empty_scope(user_id, monkeypatch):
    _write_index(user_id, _unit_rows(2))
    monkeypatch.setattr(rag, "_embedder", _FakeEmbedder([1.0, 0.0, 0.0, 0.0]))
    rag._index_cache.clear()

    assert rag.semantic_search(user_id, "soru", top_k=3, row_to_chunk_id={}) == []
