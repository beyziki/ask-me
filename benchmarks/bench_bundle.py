"""Frontend bundle boyutu benchmark'ı.

NE ÖLÇÜYOR
----------
`npm run build` çıktısındaki her dosyanın ham ve gzip boyutu, ayrıca CHUNK
SAYISI. Chunk sayısı boyut kadar önemli: tek bir dev chunk, kullanıcının
giriş ekranını görmek için tüm uygulamayı (react-markdown, remark-gfm dahil)
indirmesi demek (audit P2-5). Kod bölme sonrası aynı toplam boyut daha çok
chunk'a dağılır ve ilk yükleme küçülür — bu benchmark tam da o farkı gösterir.

Gzip boyutu Python'da hesaplanıyor (vite'ın çıktı formatını ayrıştırmak
yerine): sürüm değişikliklerinden etkilenmez.

    python -m benchmarks.bench_bundle
    python -m benchmarks.bench_bundle --skip-build   # mevcut dist/ üzerinden
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
import time
from pathlib import Path

from benchmarks._harness import Measurement, PROJECT_ROOT, print_table, save, skipped

FRONTEND = PROJECT_ROOT / "frontend-web"
# İlk açılışta MUTLAKA indirilen dosyalar (giriş noktası + stil).
# Lazy-loaded chunk'lar bu sayıya girmez — kod bölmenin kazancı burada görünür.
_ENTRY_HINTS = ("index-", "main-")


def _gzip_size(path: Path) -> int:
    return len(gzip.compress(path.read_bytes(), compresslevel=9))


def run(skip_build: bool = False):
    measurements: list[Measurement] = []
    extra: dict = {}

    if not FRONTEND.exists():
        return [skipped("tumu", f"{FRONTEND} yok")], extra

    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm is None:
        return [skipped("tumu", "npm bulunamadı")], extra

    if not (FRONTEND / "node_modules").exists():
        return [skipped("tumu", "node_modules yok — önce `npm install` çalıştırın")], extra

    dist = FRONTEND / "dist"

    if not skip_build:
        if dist.exists():
            shutil.rmtree(dist)
        start = time.perf_counter()
        proc = subprocess.run(
            [npm, "run", "build"], cwd=FRONTEND, capture_output=True, text=True
        )
        build_ms = (time.perf_counter() - start) * 1000
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
            return [skipped("tumu", "npm run build başarısız: " + " | ".join(tail))], extra
        measurements.append(
            Measurement(name="build_suresi", unit="ms", samples=[build_ms],
                        note="tsc -b && vite build")
        )

    if not dist.exists():
        return measurements + [skipped("bundle", "dist/ yok")], extra

    files = []
    for path in sorted(dist.rglob("*")):
        if not path.is_file():
            continue
        raw = path.stat().st_size
        gz = _gzip_size(path)
        rel = path.relative_to(dist).as_posix()
        files.append({"file": rel, "raw_bytes": raw, "gzip_bytes": gz,
                      "ext": path.suffix.lstrip(".")})

    js = [f for f in files if f["ext"] == "js"]
    css = [f for f in files if f["ext"] == "css"]
    entry = [f for f in js + css if any(h in Path(f["file"]).name for h in _ENTRY_HINTS)]

    extra["files"] = files
    extra["js_chunk_count"] = len(js)
    extra["css_file_count"] = len(css)
    extra["code_splitting_active"] = len(js) > 1

    def kb(total_bytes: int) -> float:
        return total_bytes / 1024

    measurements += [
        Measurement(name="js_chunk_sayisi", unit="adet", samples=[len(js)],
                    note="1 ise kod bölme YOK (audit P2-5)"),
        Measurement(name="toplam_js_gzip", unit="kB",
                    samples=[kb(sum(f["gzip_bytes"] for f in js))]),
        Measurement(name="toplam_css_gzip", unit="kB",
                    samples=[kb(sum(f["gzip_bytes"] for f in css))]),
        Measurement(
            name="ilk_yukleme_gzip",
            unit="kB",
            samples=[kb(sum(f["gzip_bytes"] for f in (entry or js + css)))],
            note="giriş ekranını görmek için indirilmesi ZORUNLU olan miktar — "
                 "asıl optimize edilecek sayı",
        ),
        Measurement(name="toplam_dist_gzip", unit="kB",
                    samples=[kb(sum(f["gzip_bytes"] for f in files))]),
    ]
    return measurements, extra


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    measurements, extra = run(skip_build=args.skip_build)
    print_table("bundle", measurements)

    if extra.get("files"):
        print(f"\n  {'dosya':<40} {'ham':>10} {'gzip':>10}")
        for f in sorted(extra["files"], key=lambda x: -x["gzip_bytes"])[:12]:
            print(f"  {f['file']:<40} {f['raw_bytes']/1024:>9.1f}k {f['gzip_bytes']/1024:>9.1f}k")
        if not extra.get("code_splitting_active"):
            print("\n  ! Tek JS chunk'ı var: kod bölme yok. Giriş ekranı bile tüm "
                  "uygulamayı\n    (react-markdown + remark-gfm dahil) indiriyor — audit P2-5.")

    path = save("bundle", measurements, extra)
    print(f"\n  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
