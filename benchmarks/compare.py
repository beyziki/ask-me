"""İki benchmark çalıştırmasını yan yana koyar: ÖNCE / SONRA.

Her optimizasyonun kuralı bu: değişiklikten önce ve sonra aynı benchmark
çalıştırılır, fark buradan okunur. "Daha hızlı hissettiriyor" bir kanıt değil.

    python -m benchmarks.run_all --label baseline      # değişiklikten ÖNCE
    ... değişikliği yap ...
    python -m benchmarks.run_all --label faz2          # değişiklikten SONRA
    python -m benchmarks.compare baseline faz2

GÜRÜLTÜ EŞİĞİ
-------------
%5'ten küçük farklar "değişmedi" sayılıyor: ölçüm gürültüsü bu mertebede ve
her küçük dalgalanmayı "iyileşme" diye raporlamak yanıltıcı olurdu.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmarks._harness import RESULTS_DIR

NOISE_THRESHOLD_PCT = 5.0


def _load(name: str) -> dict:
    path = Path(name)
    if not path.exists():
        path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"Bulunamadı: {name} (ne dosya ne de {RESULTS_DIR}/{name}.json)")
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten(payload: dict) -> dict[str, dict]:
    """Hem `run_all` özetini hem tek suite çıktısını aynı biçime indirger."""
    out: dict[str, dict] = {}
    if "suites" in payload:
        for suite, data in payload["suites"].items():
            for m in data.get("measurements", []):
                out[f"{suite}.{m['name']}"] = m
    else:
        suite = payload.get("suite", "?")
        for m in payload.get("measurements", []):
            out[f"{suite}.{m['name']}"] = m
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()

    before = _flatten(_load(args.before))
    after = _flatten(_load(args.after))

    keys = sorted(set(before) | set(after))
    name_width = max((len(k) for k in keys), default=20)

    print(f"\n{'=' * (name_width + 46)}")
    print(f"{'ölçüm':<{name_width}}  {'ÖNCE':>11}  {'SONRA':>11}  {'fark':>12}")
    print(f"{'=' * (name_width + 46)}")

    improved, regressed, unchanged = [], [], []

    for key in keys:
        b, a = before.get(key), after.get(key)
        b_val = b.get("median") if b else None
        a_val = a.get("median") if a else None
        unit = (a or b or {}).get("unit", "ms")

        if b_val is None and a_val is None:
            print(f"{key:<{name_width}}  {'atlandı':>11}  {'atlandı':>11}  {'—':>12}")
            continue
        if b_val is None:
            print(f"{key:<{name_width}}  {'—':>11}  {a_val:>10.2f}{unit[:1]}  {'YENİ':>12}")
            continue
        if a_val is None:
            print(f"{key:<{name_width}}  {b_val:>10.2f}{unit[:1]}  {'—':>11}  {'KAYIP':>12}")
            continue

        delta_pct = ((a_val - b_val) / b_val * 100) if b_val else 0.0
        if abs(delta_pct) < NOISE_THRESHOLD_PCT:
            marker, bucket = "≈", unchanged
        elif delta_pct < 0:
            marker, bucket = "↓", improved
        else:
            marker, bucket = "↑", regressed
        bucket.append((key, b_val, a_val, delta_pct))

        print(
            f"{key:<{name_width}}  {b_val:>10.2f}{unit[:1]}  {a_val:>10.2f}{unit[:1]}  "
            f"{marker} {delta_pct:>+8.1f}%"
        )

    print(f"{'=' * (name_width + 46)}")
    print(f"  iyileşen: {len(improved)}   kötüleşen: {len(regressed)}   "
          f"değişmeyen (<%{NOISE_THRESHOLD_PCT:.0f}): {len(unchanged)}")

    if regressed:
        print("\n  DİKKAT — kötüleşenler:")
        for key, b_val, a_val, pct in sorted(regressed, key=lambda x: -x[3]):
            print(f"    {key}: {b_val:.2f} -> {a_val:.2f} ({pct:+.1f}%)")

    # Kötüleşme varsa çıkış kodu 1: CI'da regresyonu kırmızıya çevirmek için.
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())
