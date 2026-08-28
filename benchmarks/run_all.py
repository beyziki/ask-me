"""Çalıştırılabilen tüm benchmark'ları koşup tek bir özet üretir.

HER SUITE AYRI BİR SÜREÇTE ÇALIŞIR — bu bilinçli ve gerekli:

  * `resources` suite'i kademeli RAM ölçüyor (import -> model -> korpus ->
    önbellek). Aynı süreçte ondan önce `retrieval` çalışmışsa korpus ZATEN
    yüklü olur ve tüm deltalar 0 çıkar — ölçüm sessizce anlamsızlaşır.
    (Bu hata gerçekten yapıldı ve ölçümde yakalandı.)
  * `llm` suite'i `lru_cache`'li `_get_manager`/`_get_model_id` üzerinden
    SOĞUK warmup ölçüyor; aynı süreçte ikinci kez çalıştırılamaz.
  * Bir suite çökerse diğerlerini etkilemez.

Ortam eksikse (Foundry kapalı, backend ayakta değil, embedding modeli yok)
suite "atlandı" olarak işaretlenir, koşu devam eder.

    python -m benchmarks.run_all
    python -m benchmarks.run_all --only retrieval,resources
    python -m benchmarks.run_all --label baseline
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

from benchmarks._harness import RESULTS_DIR, environment

# Sıra önemli: ucuz ve bağımsız olanlar önce, ortam gerektirenler sonra.
SUITES: dict[str, tuple[str, str]] = {
    "retrieval": ("benchmarks.bench_retrieval", "Retrieval (DB + BM25 + FAISS)"),
    "resources": ("benchmarks.bench_resources", "RAM / VRAM / önbellek"),
    "ingestion": ("benchmarks.bench_ingestion", "Ingestion (parse + chunk + embed)"),
    "bundle": ("benchmarks.bench_bundle", "Frontend bundle boyutu"),
    "llm": ("benchmarks.bench_llm", "LLM (warmup, TTFT, üretim) — Foundry Local gerekir"),
    "concurrent": ("benchmarks.bench_concurrent", "Eşzamanlılık — çalışan backend gerekir"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="Virgülle ayrılmış suite adları")
    parser.add_argument("--label", default="", help="Sonucu ayrıca bu adla kaydet (ör. baseline)")
    parser.add_argument("--timeout", type=int, default=3600, help="Suite başına saniye")
    args = parser.parse_args()

    wanted = [s.strip() for s in args.only.split(",") if s.strip()] or list(SUITES)
    unknown = [s for s in wanted if s not in SUITES]
    if unknown:
        print(f"Bilinmeyen suite: {', '.join(unknown)}")
        print(f"Mevcut: {', '.join(SUITES)}")
        return 1

    summary: dict = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment(),
        "suites": {},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name in wanted:
        module_name, title = SUITES[name]
        print(f"\n\n{'#' * 78}\n# {title}\n{'#' * 78}", flush=True)

        # Ayrı süreç: yukarıdaki modül docstring'indeki gerekçe.
        proc = subprocess.run(
            [sys.executable, "-m", module_name],
            timeout=args.timeout,
        )

        result_file = RESULTS_DIR / f"{name}.json"
        if proc.returncode != 0:
            summary["suites"][name] = {
                "ok": False,
                "error": f"çıkış kodu {proc.returncode}",
            }
            continue
        if not result_file.exists():
            summary["suites"][name] = {"ok": False, "error": "sonuç dosyası üretilmedi"}
            continue

        payload = json.loads(result_file.read_text(encoding="utf-8"))
        summary["suites"][name] = {
            "ok": True,
            "measurements": payload.get("measurements", []),
            "extra": payload.get("extra", {}),
        }

    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.label:
        (RESULTS_DIR / f"{args.label}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # --- Kapanış özeti ----------------------------------------------------
    print(f"\n\n{'=' * 78}\nKOŞU ÖZETİ\n{'=' * 78}")
    for name in wanted:
        data = summary["suites"].get(name, {})
        if not data.get("ok"):
            print(f"  {name:<12} BAŞARISIZ — {data.get('error', '?')}")
            continue
        measured = [m for m in data["measurements"] if not m.get("skipped_reason")]
        skipped = [m for m in data["measurements"] if m.get("skipped_reason")]
        print(f"  {name:<12} {len(measured)} ölçüm, {len(skipped)} atlandı")

    print(f"\nÖzet: {RESULTS_DIR / 'summary.json'}")
    if args.label:
        print(f"Etiketli kopya: {RESULTS_DIR / (args.label + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
