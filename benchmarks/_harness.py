"""Benchmark'lar için ortak altyapı: zamanlama, istatistik, JSON çıktı.

TASARIM KURALLARI
-----------------
1. **Hiçbir benchmark üretim kodunu değiştirmez.** Hepsi `backend.app...`
   modüllerini olduğu gibi import edip çağırır. Bir benchmark'ın "düzeltme"
   yapması gerekiyorsa, o düzeltme üretim koduna aittir.
2. **Hiçbir benchmark üretim verisine YAZMAZ.** Gerçek `data/ask_me.db`
   üzerinde ölçüm yapanlar dosyayı geçici bir kopyaya alır.
3. **Çıktı makine-okunur.** Her benchmark `benchmarks/results/*.json`
   üretir; `compare.py` iki çalıştırmayı yan yana koyar. "Önce/sonra"
   karşılaştırması buna dayanıyor.
4. **Windows'ta çalışır.** POSIX'e özgü hiçbir şey kullanılmıyor;
   `psutil`/`nvidia-smi` yoksa o metrik `null` geçilir, benchmark çökmez.

KULLANIM
--------
    python -m benchmarks.run_all                # çalıştırılabilen her şey
    python -m benchmarks.bench_retrieval        # tek bir benchmark
    python -m benchmarks.compare baseline.json son.json
"""
from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --- Ölçüm ----------------------------------------------------------------


@dataclass
class Measurement:
    """Tek bir ölçümün sonucu.

    Ortalama yerine MEDYAN ve p95 raporlanıyor: latency dağılımları çarpık
    (birkaç yavaş örnek ortalamayı yanıltıcı biçimde yukarı çekiyor) ve
    kullanıcının hissettiği şey medyan ile kuyruk.
    """

    name: str
    unit: str = "ms"
    samples: list[float] = field(default_factory=list)
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    skipped_reason: str | None = None

    @property
    def median(self) -> float | None:
        return statistics.median(self.samples) if self.samples else None

    @property
    def p95(self) -> float | None:
        if not self.samples:
            return None
        if len(self.samples) < 20:
            return max(self.samples)
        ordered = sorted(self.samples)
        return ordered[int(len(ordered) * 0.95)]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["median"] = self.median
        data["p95"] = self.p95
        data["min"] = min(self.samples) if self.samples else None
        data["max"] = max(self.samples) if self.samples else None
        data["n"] = len(self.samples)
        # Ham örnekleri saklıyoruz: sonradan farklı bir istatistik hesaplamak
        # istersek benchmark'ı yeniden çalıştırmak gerekmesin.
        return data


def timed(
    name: str,
    fn: Callable[[], Any],
    *,
    repeat: int = 5,
    warmup: int = 1,
    unit: str = "ms",
    note: str = "",
) -> Measurement:
    """`fn`'i `repeat` kez çalıştırıp süreleri toplar.

    `warmup`: ilk çağrı(lar) ölçüme DAHİL EDİLMEZ. Python'da ilk çağrı
    import/JIT/cache ısınması taşır ve bu, ölçmek istediğimiz kararlı hâli
    temsil etmez. (Soğuk başlangıcı ölçmek istediğimiz yerlerde — ör. model
    warmup — `warmup=0` veriyoruz ve bunu `note`'ta belirtiyoruz.)
    """
    m = Measurement(name=name, unit=unit, note=note)
    try:
        for _ in range(warmup):
            fn()
        for _ in range(repeat):
            start = time.perf_counter()
            fn()
            m.samples.append((time.perf_counter() - start) * 1000.0)
    except Exception as exc:  # benchmark asla suite'i çökertmemeli
        m.skipped_reason = f"{type(exc).__name__}: {exc}"
    return m


def skipped(name: str, reason: str, unit: str = "ms") -> Measurement:
    return Measurement(name=name, unit=unit, skipped_reason=reason)


# --- Ortam bilgisi --------------------------------------------------------
# Sayılar ancak hangi makinede alındığı bilinirse karşılaştırılabilir.


def _gpu_info() -> dict[str, Any] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        name, total, used, driver = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        return {
            "name": name,
            "vram_total_mb": int(total),
            "vram_used_mb": int(used),
            "driver": driver,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "processor": platform.processor() or platform.machine(),
        "logical_cores": None,
        "physical_cores": None,
        "ram_total_mb": None,
    }
    try:
        import psutil

        info["logical_cores"] = psutil.cpu_count(logical=True)
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_total_mb"] = round(psutil.virtual_memory().total / 1024 / 1024)
    except ImportError:
        pass
    return info


def process_rss_mb() -> float | None:
    """Bu sürecin o anki bellek kullanımı (MB). psutil yoksa None."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return None


def gpu_used_mb() -> int | None:
    info = _gpu_info()
    return info["vram_used_mb"] if info else None


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": _cpu_info(),
        "gpu": _gpu_info(),
    }
    try:
        from backend.app.core.config import settings

        env["settings"] = {
            "foundry_model_alias": settings.foundry_model_alias,
            "model_has_thinking": settings.model_has_thinking,
            "answer_token_budgets": settings.answer_token_budgets,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k_semantic": settings.top_k_semantic,
            "top_k_bm25": settings.top_k_bm25,
            "max_context_chunks": settings.max_context_chunks,
            "max_context_chars": settings.max_context_chars,
            "min_relevance_score": settings.min_relevance_score,
            "rrf_k": settings.rrf_k,
        }
    except Exception as exc:
        env["settings_error"] = str(exc)
    return env


# --- Rapor ----------------------------------------------------------------


def save(suite: str, measurements: list[Measurement], extra: dict[str, Any] | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": suite,
        "environment": environment(),
        "extra": extra or {},
        "measurements": [m.to_dict() for m in measurements],
    }
    path = RESULTS_DIR / f"{suite}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def print_table(suite: str, measurements: list[Measurement]) -> None:
    print(f"\n{'=' * 78}\n{suite}\n{'=' * 78}")
    width = max((len(m.name) for m in measurements), default=10)
    for m in measurements:
        if m.skipped_reason:
            print(f"  {m.name:<{width}}  ATLANDI — {m.skipped_reason}")
            continue
        median = m.median
        p95 = m.p95
        line = f"  {m.name:<{width}}  {median:>10.2f} {m.unit}"
        if p95 is not None and len(m.samples) > 1:
            line += f"   (p95 {p95:.2f}, n={len(m.samples)})"
        print(line)
        if m.note:
            print(f"  {'':<{width}}  ↳ {m.note}")


def temp_copy_of_production_db(dest_dir: Path) -> Path | None:
    """Üretim veritabanının geçici bir kopyasını üretir.

    Benchmark'lar gerçek korpus üzerinde ölçüm yapmalı (sentetik veri
    yanıltıcı olurdu) ama ona ASLA yazmamalı.
    """
    import shutil

    src = PROJECT_ROOT / "data" / "ask_me.db"
    if not src.exists():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "ask_me_bench.db"
    shutil.copy2(src, dest)
    return dest
