"""Kaynak kullanımı benchmark'ı: RAM, VRAM ve önbellek büyümesi.

NE ÖLÇÜYOR
----------
1. **Kademeli RAM** — backend'in belleği nereye gidiyor: import, embedding
   modeli, BM25 önbelleği, FAISS index. Her adım ayrı ölçülüyor ki "backend
   1.5 GB RAM yiyor" gibi eyleme geçirilemez bir cümle yerine hangi bileşenin
   ne kadar tuttuğu bilinsin.
2. **İstek başına geçici RAM** — `_retrieve` her soruda kullanıcının TÜM
   chunk metnini belleğe çekiyor (audit P1-4). Eşzamanlı istek sayısıyla
   çarpıldığında OOM riski buradan geliyor.
3. **Önbellek sızıntısı** — `rag._index_cache` sınırsız büyüyor ve dosya
   değiştiğinde eski girdiyi bırakmıyor (audit P1-5). Bu ölçüm, tekrarlanan
   "yükleme" senaryosunda önbelleğin gerçekten büyüdüğünü kanıtlıyor.
4. **VRAM** — model yüklü/yüksüz fark (nvidia-smi varsa).
"""
from __future__ import annotations

import gc
import sqlite3
import sys
import tempfile
from pathlib import Path

from benchmarks._harness import (
    Measurement,
    gpu_used_mb,
    print_table,
    process_rss_mb,
    save,
    skipped,
    temp_copy_of_production_db,
)


def _rss() -> float:
    gc.collect()
    value = process_rss_mb()
    return value if value is not None else float("nan")


def _point(name: str, mb: float, note: str = "") -> Measurement:
    return Measurement(name=name, unit="MB", samples=[mb], note=note)


def run():
    measurements: list[Measurement] = []
    extra: dict = {"vram_before": gpu_used_mb()}

    if process_rss_mb() is None:
        return [skipped("tumu", "psutil kurulu değil — `pip install psutil`")], extra

    baseline = _rss()
    measurements.append(_point("rss_baslangic", baseline, "yalnızca Python + benchmark modülleri"))

    # --- 1) Backend modüllerini import et --------------------------------
    from backend.app.core.config import settings  # noqa: F401
    from backend.app.services import rag

    after_import = _rss()
    measurements.append(
        _point("rss_backend_import_sonrasi", after_import,
               f"+{after_import - baseline:.0f} MB — faiss, sentence_transformers, sqlalchemy")
    )

    # --- 2) Embedding modeli ---------------------------------------------
    try:
        rag.get_embedder()
        after_embedder = _rss()
        measurements.append(
            _point("rss_embedding_modeli_sonrasi", after_embedder,
                   f"+{after_embedder - after_import:.0f} MB — {settings.embedding_model}")
        )
    except Exception as exc:
        after_embedder = after_import
        measurements.append(skipped("rss_embedding_modeli_sonrasi", f"{type(exc).__name__}: {exc}"))

    # --- 3) Gerçek korpusu belleğe al (bir isteğin yaptığı iş) -----------
    tmp_dir = Path(tempfile.mkdtemp(prefix="ask-me-bench-res-"))
    db_path = temp_copy_of_production_db(tmp_dir)
    if db_path is None:
        measurements.append(skipped("rss_korpus_yuklendikten_sonra", "data/ask_me.db yok"))
        return measurements, extra

    conn = sqlite3.connect(str(db_path))
    owner_row = conn.execute(
        "SELECT owner_id, COUNT(*) c FROM chunks GROUP BY owner_id ORDER BY c DESC LIMIT 1"
    ).fetchone()
    if owner_row is None:
        conn.close()
        measurements.append(skipped("rss_korpus_yuklendikten_sonra", "chunk yok"))
        return measurements, extra
    owner_id, chunk_count = owner_row

    rows = conn.execute(
        "SELECT id, content FROM chunks WHERE owner_id = ?", (owner_id,)
    ).fetchall()
    conn.close()
    chunk_id_to_text = {r[0]: r[1] for r in rows}

    after_corpus = _rss()
    measurements.append(
        _point("rss_korpus_yuklendikten_sonra", after_corpus,
               f"+{after_corpus - after_embedder:.0f} MB — {chunk_count} chunk, "
               f"api/ask.py:121 bunu HER SORUDA yapıyor")
    )
    extra["per_request_corpus_mb"] = round(after_corpus - after_embedder, 1)
    extra["chunk_count"] = chunk_count
    extra["projected_rss_at_5_concurrent_mb"] = round(
        after_corpus + 4 * (after_corpus - after_embedder), 1
    )

    # --- 4) BM25 önbelleği -----------------------------------------------
    try:
        rag.bm25_search(chunk_id_to_text, "Turing makinesi", settings.top_k_bm25)
        after_bm25 = _rss()
        measurements.append(
            _point("rss_bm25_onbellegi_sonrasi", after_bm25,
                   f"+{after_bm25 - after_corpus:.0f} MB — rag.py:_bm25_cache")
        )
    except Exception as exc:
        after_bm25 = after_corpus
        measurements.append(skipped("rss_bm25_onbellegi_sonrasi", f"{type(exc).__name__}: {exc}"))

    # --- 5) FAISS index önbelleğinin gerçek büyüme sınırı ----------------
    #
    # DÜZELTME (2026-08-17, ölçümle): Audit raporunda P1-5, "aynı dosyaya her
    # yazmada yeni bir önbellek girdisi kalıyor, sınırsız sızıntı" diye
    # yazılmıştı. BU YANLIŞ. `_read_index_cached` (rag.py:76) önbellek
    # anahtarı olarak `str(path)` kullanıyor — `(mtime, size)` yalnızca
    # DEĞER'in bir parçası. Aynı dosyaya tekrar yazıldığında girdi
    # ÜZERİNE YAZILIYOR, birikmiyor. Aşağıdaki ölçüm bunu doğruluyor.
    #
    # GERÇEK durum: önbellek KULLANICI SAYISI kadar büyüyor (her kullanıcının
    # kendi index dosyası var) ve hiçbir üst sınırı yok. Bu bir "sızıntı"
    # değil, sınırsız bir önbellek — ama 100 kullanıcılı bir kurulumda 100
    # FAISS index'i aynı anda bellekte tutuluyor demek. Aşağıda ölçülen şey
    # bu: kullanıcı başına maliyet ve toplam büyüme.
    import faiss
    import numpy as np

    cache_dir = tmp_dir / "cache_growth"
    cache_dir.mkdir(exist_ok=True)

    with rag._index_cache_lock:
        rag._index_cache.clear()
    before_growth = _rss()

    vectors = np.random.rand(2000, 384).astype("float32")
    faiss.normalize_L2(vectors)

    # (a) TEK dosyaya 10 kez yaz — birikiyor mu?
    single_path = cache_dir / "user_single.faiss"
    for i in range(1, 11):
        index = faiss.IndexFlatIP(384)
        index.add(vectors[: i * 200])
        faiss.write_index(index, str(single_path))
        rag._read_index_cached(single_path)
    with rag._index_cache_lock:
        entries_single = len(rag._index_cache)

    # (b) 10 FARKLI kullanıcı dosyası — asıl büyüme ekseni
    for user in range(10):
        path = cache_dir / f"user_{user}.faiss"
        index = faiss.IndexFlatIP(384)
        index.add(vectors)
        faiss.write_index(index, str(path))
        rag._read_index_cached(path)
    with rag._index_cache_lock:
        entries_multi = len(rag._index_cache)
    after_growth = _rss()

    measurements.append(
        _point("rss_10_kullanici_index_onbellegi", after_growth,
               f"+{after_growth - before_growth:.0f} MB — önbellekte {entries_multi} girdi, "
               f"üst sınır YOK (kullanıcı sayısıyla büyür)")
    )
    extra["index_cache_entries_after_10_writes_same_file"] = entries_single
    extra["index_cache_entries_after_10_distinct_users"] = entries_multi
    extra["index_cache_has_size_limit"] = False
    extra["audit_p1_5_correction"] = (
        "Aynı dosyaya tekrar yazmak önbellekte BİRİKMİYOR (anahtar = dosya yolu); "
        f"10 yazma sonrası girdi sayısı: {entries_single}. Gerçek risk, kullanıcı "
        f"sayısıyla sınırsız büyüme: 10 kullanıcı -> {entries_multi} girdi."
    )

    with rag._index_cache_lock:
        rag._index_cache.clear()

    extra["vram_after"] = gpu_used_mb()
    return measurements, extra


def main() -> int:
    measurements, extra = run()
    print_table("resources", measurements)

    if extra.get("audit_p1_5_correction"):
        print(f"\n  Audit P1-5 DÜZELTMESİ: {extra['audit_p1_5_correction']}")
    if extra.get("per_request_corpus_mb"):
        print(
            f"\n  İstek başına geçici RAM (audit P1-4): {extra['per_request_corpus_mb']} MB\n"
            f"  5 eşzamanlı istekte tahmini RSS: {extra['projected_rss_at_5_concurrent_mb']} MB"
        )

    path = save("resources", measurements, extra)
    print(f"\n  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
