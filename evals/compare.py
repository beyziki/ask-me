"""İki eval çalıştırmasını karşılaştırır: hangi soru düzeldi, hangisi bozuldu.

Toplam metriklerdeki değişim tek başına yeterli değil: recall %70'ten %72'ye
çıkarken 3 soru düzelip 2 soru bozulmuş olabilir ve bozulanlar önemli olanlar
olabilir. Bu yüzden vaka vaka fark da yazdırılıyor — REGRESYON, ortalamanın
arkasına saklanamasın.

    python -m evals.run_eval --label baseline
    ... değişikliği yap ...
    python -m evals.run_eval --label faz2
    python -m evals.compare baseline faz2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

_HEADLINE = [
    ("raw_recall_at_k", "ham recall@k", True),
    ("raw_precision_at_k", "ham precision@k", True),
    ("raw_mrr", "ham MRR", True),
    ("delivered_recall_at_k", "teslim recall@k", True),
    ("delivered_precision_at_k", "teslim precision@k", True),
    ("delivered_mrr", "teslim MRR", True),
    ("term_hit_rate", "terim isabeti", True),
    ("has_context_accuracy", "has_context doğruluğu", True),
    ("false_no_context_rate", "yanlış 'bulunamadı'", False),
    ("false_has_context_rate", "yanlış 'bulundu'", False),
    ("median_retrieve_latency_ms", "medyan _retrieve (ms)", False),
]


def _load(name: str) -> dict:
    path = Path(name)
    if not path.exists():
        path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"Bulunamadı: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()

    before, after = _load(args.before), _load(args.after)
    b_sum, a_sum = before["summary"], after["summary"]

    print(f"\n{'=' * 76}")
    print(f"{'metrik':<28} {'ÖNCE':>12} {'SONRA':>12} {'fark':>14}")
    print(f"{'=' * 76}")
    for key, label, higher_is_better in _HEADLINE:
        b, a = b_sum.get(key), a_sum.get(key)
        if b is None or a is None:
            print(f"{label:<28} {'—':>12} {'—':>12} {'—':>14}")
            continue
        delta = a - b
        if abs(delta) < 1e-9:
            marker = "≈"
        elif (delta > 0) == higher_is_better:
            marker = "İYİ"
        else:
            marker = "KÖTÜ"
        is_ms = key.endswith("_ms")
        fmt = (lambda v: f"{v:>11.0f}m") if is_ms else (lambda v: f"{v * 100:>11.1f}%")
        print(f"{label:<28} {fmt(b)} {fmt(a)} {marker:>6} {delta * (1 if is_ms else 100):>+7.1f}")

    # --- Vaka bazlı fark --------------------------------------------------
    b_cases = {c["id"]: c for c in before["cases"]}
    a_cases = {c["id"]: c for c in after["cases"]}

    fixed, broken = [], []
    for case_id in sorted(set(b_cases) & set(a_cases)):
        b, a = b_cases[case_id], a_cases[case_id]
        for metric in ("delivered_recall_at_k", "term_hit", "has_context_correct"):
            bv, av = b.get(metric), a.get(metric)
            if bv is None or av is None:
                continue
            if not bv and av:
                fixed.append((case_id, metric))
            elif bv and not av:
                broken.append((case_id, metric))

    print(f"\n{'=' * 76}")
    if fixed:
        print(f"  DÜZELEN ({len(fixed)}):")
        for case_id, metric in fixed:
            print(f"    + {case_id:<24} {metric}")
    if broken:
        print(f"\n  BOZULAN ({len(broken)}) — DİKKAT:")
        for case_id, metric in broken:
            print(f"    - {case_id:<24} {metric}")
    if not fixed and not broken:
        print("  Vaka bazında değişiklik yok.")

    # Regresyon varsa çıkış kodu 1 — CI'da kırmızıya çevirmek için.
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
