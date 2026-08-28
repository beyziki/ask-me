"""Doküman ingestion benchmark'ı: PDF parse -> chunking -> embedding -> FAISS.

NE ÖLÇÜYOR
----------
`POST /documents/upload`'ın içindeki dört adımın her birini AYRI AYRI. Ayrı
ölçmek şart: "yükleme yavaş" tek başına eyleme geçirilebilir bir bilgi değil;
sürenin pypdf'te mi, embedding'de mi, yoksa FAISS yazmada mı geçtiğini bilmek
hangi optimizasyonun işe yarayacağını belirliyor.

Gerçek dosyalar `data/uploads/` altından okunuyor (sentetik metin, PDF parse
maliyetini ve chunk dağılımını temsil etmezdi). Hiçbir şey yazılmıyor:
FAISS index geçici bir dizine kuruluyor.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from benchmarks._harness import (
    Measurement,
    PROJECT_ROOT,
    print_table,
    process_rss_mb,
    save,
    skipped,
    timed,
)

# Boyut sınıflarına göre en fazla kaç dosya ölçülsün (her sınıftan 1).
# Amaç, dosya boyutuyla sürenin nasıl ölçeklendiğini görmek.
_SIZE_BUCKETS = [
    ("kucuk", 0, 200_000),
    ("orta", 200_000, 1_500_000),
    ("buyuk", 1_500_000, 10**12),
]


def _pick_sample_files() -> list[tuple[str, Path]]:
    uploads = PROJECT_ROOT / "data" / "uploads"
    if not uploads.exists():
        return []
    candidates = sorted(
        (p for p in uploads.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.stat().st_size,
    )
    picked: list[tuple[str, Path]] = []
    for label, low, high in _SIZE_BUCKETS:
        for path in candidates:
            size = path.stat().st_size
            if low <= size < high:
                picked.append((label, path))
                break
    return picked


def run():
    from backend.app.core.config import settings
    from backend.app.services import rag
    from backend.app.services.ingestion import chunk_text, extract_text

    measurements: list[Measurement] = []
    extra: dict = {"files": []}

    samples = _pick_sample_files()
    if not samples:
        return [skipped("tumu", "data/uploads/ altında PDF bulunamadı")], extra

    embedder = None
    try:
        embedder = rag.get_embedder()
    except Exception as exc:
        extra["embedder_error"] = f"{type(exc).__name__}: {exc}"

    tmp_dir = Path(tempfile.mkdtemp(prefix="ask-me-bench-ingest-"))

    for label, path in samples:
        size_kb = round(path.stat().st_size / 1024)
        tag = f"{label}_{size_kb}kb"

        # --- 1) PDF parse -------------------------------------------------
        measurements.append(
            timed(
                f"pdf_parse_{tag}",
                lambda p=path: extract_text(p, "pdf"),
                repeat=3,
                warmup=0,  # dosya sistemi önbelleği zaten gerçek davranışın parçası
                note=f"ingestion.py:34 (pypdf) — {path.name}",
            )
        )

        text = extract_text(path, "pdf")

        # --- 2) Chunking --------------------------------------------------
        measurements.append(
            timed(
                f"chunking_{tag}",
                lambda t=text: chunk_text(t),
                repeat=5,
                note=f"ingestion.py:41 — chunk_size={settings.chunk_size} kelime",
            )
        )

        chunks = chunk_text(text)
        lengths = [len(c) for c in chunks]
        file_info = {
            "name": path.name,
            "bucket": label,
            "size_kb": size_kb,
            "text_chars": len(text),
            "chunk_count": len(chunks),
            "chunk_chars_min": min(lengths) if lengths else 0,
            "chunk_chars_median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
            "chunk_chars_max": max(lengths) if lengths else 0,
            # Boş/çöp chunk'lar FAISS'e de yazılıyor (bkz. audit P0-5).
            "empty_chunks": sum(1 for c in chunks if not c.strip()),
            "tiny_chunks_under_50_chars": sum(1 for c in chunks if len(c.strip()) < 50),
        }
        extra["files"].append(file_info)

        # --- 3) Embedding -------------------------------------------------
        if embedder is None:
            measurements.append(skipped(f"embedding_{tag}", extra.get("embedder_error", "model yok")))
            measurements.append(skipped(f"faiss_add_write_{tag}", extra.get("embedder_error", "model yok")))
            continue

        measurements.append(
            timed(
                f"embedding_{tag}",
                lambda c=chunks: embedder.encode(c, normalize_embeddings=True),
                repeat=1,
                warmup=0,  # gerçek yükleme de tek seferlik; ısıtma yanıltıcı olurdu
                note=f"rag.py:102 — {len(chunks)} chunk tek batch'te",
            )
        )

        # --- 4) FAISS ekleme + diske yazma --------------------------------
        # Üretimde `add_chunks_to_index` TÜM index'i her yüklemede yeniden
        # yazıyor (bkz. audit P1-5). Burada aynı davranışı geçici dizinde
        # ölçüyoruz.
        import faiss
        import numpy as np

        vectors = np.asarray(embedder.encode(chunks, normalize_embeddings=True), dtype="float32")
        out_path = tmp_dir / f"bench_{tag}.faiss"

        def _add_and_write(v=vectors, p=out_path):
            index = faiss.IndexFlatIP(v.shape[1])
            index.add(v)
            faiss.write_index(index, str(p))

        measurements.append(
            timed(
                f"faiss_add_write_{tag}",
                _add_and_write,
                repeat=3,
                note=f"rag.py:105-108 — {len(chunks)} vektör, TÜM index diske yazılıyor",
            )
        )

    extra["process_rss_mb_after"] = process_rss_mb()
    return measurements, extra


def main() -> int:
    measurements, extra = run()
    print_table("ingestion", measurements)
    if extra.get("files"):
        print("\n  Chunk dağılımı (audit P0-5 — güncel chunk_size ile yeniden işlenmiş hâli):")
        print(f"  {'dosya':<42} {'chunk':>6} {'min':>6} {'medyan':>7} {'max':>7} {'boş':>4} {'<50ch':>6}")
        for f in extra["files"]:
            print(
                f"  {f['name'][:42]:<42} {f['chunk_count']:>6} {f['chunk_chars_min']:>6} "
                f"{f['chunk_chars_median']:>7} {f['chunk_chars_max']:>7} "
                f"{f['empty_chunks']:>4} {f['tiny_chunks_under_50_chars']:>6}"
            )
    path = save("ingestion", measurements, extra)
    print(f"\n  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
