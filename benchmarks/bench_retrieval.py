"""Retrieval (hybrid RAG) benchmark'ı — LLM'e hiç gitmeden ölçülen her şey.

NE ÖLÇÜYOR
----------
`/ask` isteğinin ilk yarısı: DB'den chunk'ları çekmek, BM25 korpusunu
hazırlamak, FAISS'te aramak, sonuçları birleştirmek. Bu süre, kullanıcının
gördüğü **time-to-first-token**'a doğrudan ekleniyor (bkz. api/ask.py:_retrieve
— ilk SSE olayı `sources` bu adım bitmeden gönderilmiyor).

NEDEN AYRI BİR BENCHMARK
------------------------
LLM üretimi makineden makineye ve modelden modele 10 kat değişiyor; retrieval
ise deterministik ve tamamen bizim kontrolümüzde. İkisini karıştırmak, bir RAG
optimizasyonunun kazancını model gürültüsünde kaybetmek demek.

ÖNEMLİ: Bu benchmark üretim veritabanının GEÇİCİ BİR KOPYASI üzerinde çalışır
ve hiçbir şey yazmaz.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

from benchmarks._harness import (
    Measurement,
    print_table,
    process_rss_mb,
    save,
    skipped,
    temp_copy_of_production_db,
    timed,
)

# Ölçümde kullanılan sorgular. Bilinçli olarak karışık:
#   - Türkçe / İngilizce
#   - noktalamalı / noktalamasız
#   - korpusta karşılığı olan / olmayan (min_relevance_score yolunu tetikler)
QUERIES = [
    "Turing makinesi nedir?",
    "Context free grammar örnekleri",
    "TCP ve UDP arasındaki fark nedir",
    "push down automata nasıl çalışır",
    "What is a finite state machine?",
    "bugün hava nasıl",  # korpusta karşılığı YOK — eşik davranışını ölçer
]


def _load_corpus(db_path: Path, owner_id: int):
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT id, document_id, chunk_index, content, vector_row "
        "FROM chunks WHERE owner_id = ?",
        (owner_id,),
    ).fetchall()
    conn.close()
    return rows


def _pick_busiest_owner(db_path: Path) -> tuple[int, int] | None:
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT owner_id, COUNT(*) c FROM chunks GROUP BY owner_id ORDER BY c DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return (row[0], row[1]) if row else None


def run() -> list[Measurement]:
    from backend.app.core.config import settings
    from backend.app.services import rag

    measurements: list[Measurement] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="ask-me-bench-"))
    db_path = temp_copy_of_production_db(tmp_dir)

    if db_path is None:
        return [skipped("tumu", "data/ask_me.db bulunamadı — önce backend'i çalıştırıp veri yükleyin")]

    picked = _pick_busiest_owner(db_path)
    if picked is None:
        return [skipped("tumu", "Veritabanında hiç chunk yok")]
    owner_id, chunk_count = picked

    rows = _load_corpus(db_path, owner_id)
    chunk_id_to_text = {r[0]: r[3] for r in rows}
    row_to_chunk_id = {r[4]: r[0] for r in rows if r[4] is not None}
    corpus_chars = sum(len(t) for t in chunk_id_to_text.values())

    extra = {
        "owner_id": owner_id,
        "chunk_count": chunk_count,
        "corpus_chars": corpus_chars,
        "corpus_mb": round(corpus_chars / 1_000_000, 2),
        "queries": QUERIES,
    }

    # --- 1) DB: kullanıcının tüm chunk'larını çekmek ----------------------
    # `api/ask.py:_retrieve` her soruda bunu yapıyor (`query.all()`), üstelik
    # `content` sütunu dahil — yani korpusun tamamı her istekte belleğe geliyor.
    measurements.append(
        timed(
            "db_tum_chunklari_cek",
            lambda: _load_corpus(db_path, owner_id),
            repeat=5,
            note=f"api/ask.py:121 — {chunk_count} satır, {corpus_chars/1e6:.2f} MB metin, HER SORUDA",
        )
    )

    # --- 2) BM25: korpus tokenizasyonu ------------------------------------
    # `rag.py:258` — `_get_bm25` önbelleği VARSA BİLE bu satır her çağrıda
    # baştan çalışıyor, çünkü sorgu-terimi filtresi token listelerine ihtiyaç
    # duyuyor ve onları önbellekten okumuyor.
    texts = list(chunk_id_to_text.values())
    measurements.append(
        timed(
            "bm25_korpus_tokenizasyonu",
            lambda: [rag.tokenize(t) for t in texts],
            repeat=5,
            note="rag.py:258 — önbelleğe RAĞMEN her soruda tekrarlanıyor",
        )
    )

    # --- 3) BM25: index kurulumu (önbelleklenen kısım) --------------------
    corpus_tokens = [rag.tokenize(t) for t in texts]
    try:
        from rank_bm25 import BM25Okapi

        measurements.append(
            timed(
                "bm25_index_kurulumu",
                lambda: BM25Okapi(corpus_tokens),
                repeat=3,
                note="rag.py:246 — önbellekli, yalnızca ilk soruda ödeniyor",
            )
        )
    except ImportError as exc:
        measurements.append(skipped("bm25_index_kurulumu", str(exc)))

    # --- 4) BM25: uçtan uca arama (üretimdeki gerçek yol) ------------------
    for label, query in (("tr_kisa", QUERIES[0]), ("en", QUERIES[4]), ("alakasiz", QUERIES[5])):
        measurements.append(
            timed(
                f"bm25_search_{label}",
                lambda q=query: rag.bm25_search(chunk_id_to_text, q, settings.top_k_bm25),
                repeat=10,
                note=f"rag.py:254 — tokenizasyon DAHİL (gerçek istek yolu) · '{query}'",
            )
        )

    # --- 5) FAISS ---------------------------------------------------------
    index_path = settings.vectorstore_dir / f"user_{owner_id}.faiss"
    if not index_path.exists():
        measurements.append(skipped("faiss_read_index", f"{index_path} yok"))
        measurements.append(skipped("faiss_search", f"{index_path} yok"))
        measurements.append(skipped("semantic_search_uctan_uca", f"{index_path} yok"))
    else:
        import faiss

        measurements.append(
            timed(
                "faiss_read_index",
                lambda: faiss.read_index(str(index_path)),
                repeat=5,
                note="rag.py:81 — önbellekli (`_read_index_cached`), soğuk okuma maliyeti",
            )
        )

        index = faiss.read_index(str(index_path))
        extra["faiss"] = {
            "ntotal": index.ntotal,
            "dim": index.d,
            "type": type(index).__name__,
            "in_memory_mb": round(index.ntotal * index.d * 4 / 1_000_000, 2),
        }

        try:
            embedder = rag.get_embedder()
            q_vec = embedder.encode([QUERIES[0]], normalize_embeddings=True)
            import numpy as np

            q_vec = np.asarray(q_vec, dtype="float32")

            measurements.append(
                timed(
                    "faiss_search",
                    lambda: index.search(q_vec, min(settings.top_k_semantic, index.ntotal)),
                    repeat=20,
                    note=f"rag.py:206 — {index.ntotal} vektör, top_k={settings.top_k_semantic}",
                )
            )
            measurements.append(
                timed(
                    "embedding_sorgu",
                    lambda: embedder.encode([QUERIES[0]], normalize_embeddings=True),
                    repeat=10,
                    note="rag.py:203 — soru başına, önbelleklenmiyor",
                )
            )
            measurements.append(
                timed(
                    "semantic_search_uctan_uca",
                    lambda: rag.semantic_search(
                        owner_id, QUERIES[0], settings.top_k_semantic, row_to_chunk_id
                    ),
                    repeat=10,
                    note="rag.py:197 — embedding + FAISS arama + eşleme",
                )
            )
        except Exception as exc:
            reason = f"Embedding modeli kullanılamadı: {type(exc).__name__}: {exc}"
            for name in ("faiss_search", "embedding_sorgu", "semantic_search_uctan_uca"):
                measurements.append(skipped(name, reason))

    # --- 6) Doküman filtresi: kaç semantic hit hayatta kalıyor? -----------
    # Bu bir SÜRE ölçümü değil, bir DOĞRULUK ölçümü — ama retrieval'a ait
    # olduğu için burada duruyor.
    #
    # DÜZELTİLDİ (audit P0-1): `semantic_search` eskiden TÜM index'te arayıp
    # sonuçları sonradan filtreliyordu, bu yüzden dar bir doküman seçimi
    # semantic tarafı fiilen devre dışı bırakıyordu — aşağıdaki
    # `expected_surviving_hits_of_top_k` sütunu tam olarak bu kaybı ölçmek
    # için eklenmişti. Arama artık kapsamın İÇİNDE yapılıyor (bkz.
    # rag.py:_search_subset), yani seçim ne kadar dar olursa olsun semantic
    # taraf `min(top_k, kapsam)` kadar aday döndürüyor.
    #
    # Ölçüm yine de duruyor: bir REGRESYON NÖBETÇİSİ olarak. Aşağıdaki
    # "hayatta kalan" sayısı eski (bozuk) davranışın ne döndüreceğini
    # gösteriyor; sıfıra yakın değerler, filtreli aramanın yeniden global
    # aramaya dönmesi hâlinde kaybın ne kadar büyük olacağını hatırlatıyor.
    conn = sqlite3.connect(str(db_path))
    per_doc = conn.execute(
        "SELECT document_id, COUNT(*) FROM chunks WHERE owner_id = ? GROUP BY document_id",
        (owner_id,),
    ).fetchall()
    conn.close()
    extra["document_filter_dilution"] = [
        {
            "document_id": doc_id,
            "chunks": n,
            "corpus_share_pct": round(100 * n / chunk_count, 1),
            "expected_surviving_hits_of_top_k": round(settings.top_k_semantic * n / chunk_count, 2),
        }
        for doc_id, n in sorted(per_doc, key=lambda x: -x[1])
    ]

    extra["process_rss_mb_after"] = process_rss_mb()
    return measurements, extra


def main() -> int:
    result = run()
    measurements, extra = result if isinstance(result, tuple) else (result, {})
    print_table("retrieval", measurements)
    path = save("retrieval", measurements, extra)

    dilution = extra.get("document_filter_dilution")
    if dilution:
        print("\n  Doküman filtresi seyrelmesi (audit P0-1 — DÜZELTİLDİ; aşağıdaki")
        print("  sayılar eski global-arama davranışının ne kaybettirdiğini gösterir):")
        print(f"  {'doc_id':>8}  {'chunk':>6}  {'korpus %':>9}  {'top_k=8`den hayatta kalan':>26}")
        for row in dilution:
            print(
                f"  {row['document_id']:>8}  {row['chunks']:>6}  "
                f"{row['corpus_share_pct']:>8.1f}%  {row['expected_surviving_hits_of_top_k']:>26.2f}"
            )

    print(f"\n  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
