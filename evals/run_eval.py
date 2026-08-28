"""RAG retrieval doğruluk değerlendirmesi — LLM'e HİÇ GİTMEDEN.

NEDEN LLM YOK
-------------
Ölçmek istediğimiz şey "model iyi cevap yazdı mı" değil, "doğru kaynakları
buldu mu". İkisini karıştırmak iki sorun yaratır: (a) model örneklemesi
deterministik değil, aynı değişiklik iki kez farklı sonuç verir; (b) her
değerlendirme dakikalar sürer, kimse çalıştırmaz. Retrieval'ı tek başına
ölçmek deterministik ve saniyeler sürüyor — yani her değişiklikte
çalıştırılabilir.

ÖLÇÜLEN METRİKLER
-----------------
İki katman ayrı ayrı ölçülüyor; aradaki fark bilinçli:

  HAM SIRALAMA (raw_*)      : `semantic_search` + `bm25_search` + `rrf_merge`
                              çıktısı. "Arama doğru dokümanı bulabildi mi?"
  TESLİM EDİLEN (delivered_*): `_retrieve`'in kullanıcıya ulaştırdığı kaynaklar.
                              Yani ham sıralama + `min_relevance_score` eşiği
                              + `_dedupe_by_content` + `_cap_context` sonrası.

Aradaki BOŞLUK en önemli sayı: ham sıralama doğru dokümanı bulup teslim
edilen sonuç bulmuyorsa, sorun aramada değil eşikte/kırpmada demektir — ve
tamamen farklı bir düzeltme gerektirir.

  recall@k          : beklenen dokümanlardan en az biri sonuçlarda mı
  precision@k       : sonuçların kaçı beklenen dokümandan
  MRR               : ilk doğru dokümanın sırasının tersi
  term_hit          : beklenen ayırt edici terimlerden biri getirilen metinde geçiyor mu
  has_context_acc   : `has_context` bayrağı doğru mu (yanlış "bulunamadı" uyarısı ölçümü)

KULLANIM
--------
    python -m evals.run_eval
    python -m evals.run_eval --label baseline
    python -m evals.run_eval --only f-baby-only,f-tm-only     # tek vaka
    python -m evals.compare baseline faz2

ÖNEMLİ: Üretim veritabanının GEÇİCİ BİR KOPYASI üzerinde çalışır ve hiçbir
şey yazmaz. FAISS index'i yalnızca OKUNUR.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Veritabanını geçici kopyaya yönlendir --------------------------------
# `settings` import zamanında bir kez kuruluyor, bu yüzden env değişkenleri
# `backend.app...` import edilmeden ÖNCE ayarlanmalı (bkz. tests/conftest.py'deki
# aynı konudaki uzun not).
_TMP = Path(tempfile.mkdtemp(prefix="ask-me-eval-"))
_REAL_DB = PROJECT_ROOT / "data" / "ask_me.db"
_REAL_VECTORSTORE = PROJECT_ROOT / "data" / "vectorstore"

if _REAL_DB.exists():
    shutil.copy2(_REAL_DB, _TMP / "eval.db")
    os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'eval.db'}"
# Vectorstore GERÇEK dizin: semantic_search yalnızca okuyor, kopyalamak
# gereksiz I/O olurdu (index'ler yüzlerce MB olabilir).
os.environ.setdefault("VECTORSTORE_DIR", str(_REAL_VECTORSTORE))
os.environ["WARMUP_ON_STARTUP"] = "false"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"


# Doküman-filtreli vakaların id öneki. Bu vakalar audit P0-1'i ölçüyor ve
# özet tabloda AYRI raporlanıyor: filtresiz ortalamayla karıştırılırsa
# filtrenin verdiği zarar görünmez oluyor.
_FILTERED_PREFIXES = ("f-", "f2-")


def _is_filtered(case_id: str) -> bool:
    return case_id.startswith(_FILTERED_PREFIXES)


def _rank_of_first_expected(doc_sequence: list[int], expected: list[int]) -> int | None:
    for rank, doc_id in enumerate(doc_sequence, start=1):
        if doc_id in expected:
            return rank
    return None


def _evaluate_case(case: dict, k: int) -> dict:
    """Tek bir soruyu hem ham hem teslim edilen katmanda değerlendirir."""
    from sqlalchemy.orm import Session

    from backend.app.api.ask import _retrieve
    from backend.app.core.config import settings
    from backend.app.db.base import SessionLocal
    from backend.app.db.models import Chunk
    from backend.app.models.schemas import AskRequest
    from backend.app.services.rag import bm25_search, rrf_merge, semantic_search

    owner_id = case["owner_id"]
    expected = case.get("expected_documents") or []
    terms = [t.lower() for t in (case.get("required_terms") or [])]
    doc_filter = case.get("document_ids")

    result: dict = {"id": case["id"], "question": case["question"], "note": case.get("note")}

    # --- Katman 1: HAM SIRALAMA ------------------------------------------
    db: Session = SessionLocal()
    try:
        query = db.query(Chunk).filter(Chunk.owner_id == owner_id)
        if doc_filter:
            query = query.filter(Chunk.document_id.in_(doc_filter))
        chunks = query.all()
        row_to_chunk_id = {c.vector_row: c.id for c in chunks if c.vector_row is not None}
        chunk_id_to_text = {c.id: c.content for c in chunks}
        chunk_id_to_doc = {c.id: c.document_id for c in chunks}
    finally:
        db.close()

    result["corpus_chunks_in_scope"] = len(chunks)

    start = time.perf_counter()
    sem_hits = semantic_search(owner_id, case["question"], settings.top_k_semantic, row_to_chunk_id)
    bm_hits = bm25_search(chunk_id_to_text, case["question"], settings.top_k_bm25)
    merged = rrf_merge(sem_hits, bm_hits)
    result["raw_latency_ms"] = (time.perf_counter() - start) * 1000

    # Semantic'in gerçekten iş yapıp yapmadığı — audit P0-1'in doğrudan ölçümü.
    result["semantic_hit_count"] = len(sem_hits)
    result["bm25_hit_count"] = len(bm_hits)
    result["best_semantic_score"] = max((h.score for h in sem_hits), default=0.0)

    raw_docs: list[int] = []
    for hit in merged[:k]:
        doc_id = chunk_id_to_doc.get(hit.chunk_id)
        if doc_id is not None:
            raw_docs.append(doc_id)

    if expected:
        rank = _rank_of_first_expected(raw_docs, expected)
        result["raw_recall_at_k"] = 1.0 if rank else 0.0
        result["raw_precision_at_k"] = (
            sum(1 for d in raw_docs if d in expected) / len(raw_docs) if raw_docs else 0.0
        )
        result["raw_mrr"] = 1.0 / rank if rank else 0.0
    else:
        result["raw_recall_at_k"] = None
        result["raw_precision_at_k"] = None
        result["raw_mrr"] = None
    result["raw_documents"] = raw_docs

    # --- Katman 2: TESLİM EDİLEN (üretimdeki gerçek yol) -----------------
    db = SessionLocal()
    payload = AskRequest(
        question=case["question"],
        document_ids=doc_filter,
        language=case.get("language"),
    )
    start = time.perf_counter()
    context_chunks, sources, language, has_context = _retrieve(payload, owner_id, db)
    result["retrieve_latency_ms"] = (time.perf_counter() - start) * 1000

    delivered_docs = [s.document_id for s in sources]
    result["delivered_documents"] = delivered_docs
    result["has_context"] = has_context
    result["expects_context"] = case["expects_context"]
    result["has_context_correct"] = has_context == case["expects_context"]
    result["detected_language"] = language
    result["context_chunk_count"] = len(context_chunks)
    result["context_chars"] = sum(len(c) for c in context_chunks)

    if expected:
        rank = _rank_of_first_expected(delivered_docs, expected)
        result["delivered_recall_at_k"] = 1.0 if rank else 0.0
        result["delivered_precision_at_k"] = (
            sum(1 for d in delivered_docs if d in expected) / len(delivered_docs)
            if delivered_docs else 0.0
        )
        result["delivered_mrr"] = 1.0 / rank if rank else 0.0
    else:
        result["delivered_recall_at_k"] = None
        result["delivered_precision_at_k"] = None
        result["delivered_mrr"] = None

    # --- Terim kontrolü ---------------------------------------------------
    if terms:
        blob = " ".join(context_chunks).lower()
        result["term_hit"] = any(t in blob for t in terms)
        result["terms_found"] = [t for t in terms if t in blob]
    else:
        result["term_hit"] = None
        result["terms_found"] = []

    return result


def _aggregate(results: list[dict]) -> dict:
    def mean(key: str, subset: list[dict] | None = None) -> float | None:
        rows = subset if subset is not None else results
        values = [r[key] for r in rows if r.get(key) is not None]
        return statistics.mean(values) if values else None

    with_expectations = [r for r in results if r.get("raw_recall_at_k") is not None]
    filtered = [r for r in with_expectations if _is_filtered(r["id"])]
    unfiltered = [r for r in with_expectations if not _is_filtered(r["id"])]
    no_context = [r for r in results if r["expects_context"] is False]

    term_rows = [r for r in results if r.get("term_hit") is not None]

    return {
        "n_cases": len(results),
        "raw_recall_at_k": mean("raw_recall_at_k"),
        "raw_precision_at_k": mean("raw_precision_at_k"),
        "raw_mrr": mean("raw_mrr"),
        "delivered_recall_at_k": mean("delivered_recall_at_k"),
        "delivered_precision_at_k": mean("delivered_precision_at_k"),
        "delivered_mrr": mean("delivered_mrr"),
        "term_hit_rate": (
            sum(1 for r in term_rows if r["term_hit"]) / len(term_rows) if term_rows else None
        ),
        "has_context_accuracy": sum(1 for r in results if r["has_context_correct"]) / len(results),
        "false_no_context_rate": (
            sum(1 for r in with_expectations if not r["has_context"]) / len(with_expectations)
            if with_expectations else None
        ),
        "false_has_context_rate": (
            sum(1 for r in no_context if r["has_context"]) / len(no_context)
            if no_context else None
        ),
        "by_scope": {
            "tum_dosyalar": {
                "n": len(unfiltered),
                "raw_recall": mean("raw_recall_at_k", unfiltered),
                "delivered_recall": mean("delivered_recall_at_k", unfiltered),
                "semantic_hit_count": mean("semantic_hit_count", unfiltered),
            },
            "dokuman_filtreli": {
                "n": len(filtered),
                "raw_recall": mean("raw_recall_at_k", filtered),
                "delivered_recall": mean("delivered_recall_at_k", filtered),
                "semantic_hit_count": mean("semantic_hit_count", filtered),
                "_not": "audit P0-1'in ana göstergesi — semantic_hit_count düşükse filtre semantic'i kapatıyor",
            },
        },
        "median_retrieve_latency_ms": statistics.median(
            [r["retrieve_latency_ms"] for r in results]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="recall@k / precision@k için k")
    parser.add_argument("--label", default="", help="Sonucu bu adla kaydet (ör. baseline)")
    parser.add_argument("--only", default="", help="Yalnızca bu vaka id'leri (virgülle)")
    parser.add_argument("--verbose", action="store_true", help="Her vakayı tek tek yazdır")
    args = parser.parse_args()

    if not _REAL_DB.exists():
        print(f"data/ask_me.db bulunamadı ({_REAL_DB}) — önce backend'i çalıştırıp dosya yükleyin.")
        return 1

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        cases = [c for c in cases if c["id"] in wanted]
    if not cases:
        print("Çalıştırılacak vaka yok.")
        return 1

    results = []
    for case in cases:
        try:
            results.append(_evaluate_case(case, args.k))
        except Exception as exc:
            print(f"  ! {case['id']} değerlendirilemedi: {type(exc).__name__}: {exc}")
            if "sentence_transformers" in str(exc) or "403" in str(exc):
                print("    (Embedding modeli yüklenemiyor — semantic taraf ölçülemez.)")
                return 1

    summary = _aggregate(results)

    # --- Rapor ------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print(f"RAG RETRIEVAL EVAL — {len(results)} vaka, k={args.k}")
    print(f"{'=' * 100}")
    print(f"{'vaka':<22} {'kapsam':>7} {'sem':>4} {'bm25':>5} {'best_sem':>9} "
          f"{'ham':>5} {'teslim':>7} {'terim':>6} {'ctx':>5}")
    print(f"{'-' * 100}")
    for r in results:
        scope = "filtre" if _is_filtered(r["id"]) else ("yok-ctx" if not r["expects_context"] else "tümü")
        raw = "-" if r["raw_recall_at_k"] is None else ("✓" if r["raw_recall_at_k"] else "✗")
        deliv = "-" if r["delivered_recall_at_k"] is None else ("✓" if r["delivered_recall_at_k"] else "✗")
        term = "-" if r["term_hit"] is None else ("✓" if r["term_hit"] else "✗")
        ctx = "✓" if r["has_context_correct"] else "✗"
        print(
            f"{r['id']:<22} {scope:>7} {r['semantic_hit_count']:>4} {r['bm25_hit_count']:>5} "
            f"{r['best_semantic_score']:>9.3f} {raw:>5} {deliv:>7} {term:>6} {ctx:>5}"
        )

    print(f"\n{'=' * 100}\nÖZET\n{'=' * 100}")

    def pct(v):
        return "—" if v is None else f"%{v * 100:.1f}"

    print(f"  HAM SIRALAMA      recall@{args.k}={pct(summary['raw_recall_at_k'])}  "
          f"precision@{args.k}={pct(summary['raw_precision_at_k'])}  MRR={summary['raw_mrr']:.3f}")
    print(f"  TESLİM EDİLEN     recall@{args.k}={pct(summary['delivered_recall_at_k'])}  "
          f"precision@{args.k}={pct(summary['delivered_precision_at_k'])}  MRR={summary['delivered_mrr']:.3f}")
    print(f"  Terim isabeti     {pct(summary['term_hit_rate'])}")
    print(f"  has_context doğr. {pct(summary['has_context_accuracy'])}")
    print(f"    ↳ yanlış 'bulunamadı' (olması gerekirken yok): {pct(summary['false_no_context_rate'])}")
    print(f"    ↳ yanlış 'bulundu'   (olmaması gerekirken var): {pct(summary['false_has_context_rate'])}")
    print(f"  Medyan _retrieve  {summary['median_retrieve_latency_ms']:.0f} ms")

    print("\n  Kapsama göre (audit P0-1):")
    for scope, data in summary["by_scope"].items():
        print(f"    {scope:<18} n={data['n']:<3} ham recall={pct(data['raw_recall'])}  "
              f"teslim recall={pct(data['delivered_recall'])}  "
              f"ort. semantic hit={data['semantic_hit_count']:.1f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "k": args.k,
        "summary": summary,
        "cases": results,
    }
    out = RESULTS_DIR / f"{args.label or 'latest'}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> {out}")

    shutil.rmtree(_TMP, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
