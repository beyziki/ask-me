"""Eşzamanlılık benchmark'ı: N kullanıcı aynı anda soru sorunca ne oluyor?

NE ÖLÇÜYOR
----------
Çalışan bir backend'e (uvicorn) 1 / 2 / 5 / 10 eşzamanlı `/ask/stream` isteği
gönderip her birinin TTFT ve toplam süresini, hata oranını ve p95'ini ölçer.
Ayrıca yükleme sırasında `/health`'in ne kadar sürdüğünü ölçer — audit P0-3'ün
("upload event loop'u kilitliyor") doğrudan kanıtı.

NEDEN CANLI SUNUCUYA İSTEK ATIYOR (TestClient DEĞİL)
---------------------------------------------------
`TestClient` isteği aynı süreçte, ASGI'yi doğrudan çağırarak çalıştırır; uvicorn'un
event loop'unu, threadpool'unu ve gerçek soket davranışını temsil etmez. Ölçmek
istediğimiz şey tam olarak bunlar (audit P1-6: SSE'lerin threadpool worker'ı
tutması), o yüzden gerçek bir sunucu gerekiyor.

ÇALIŞTIRMA
----------
    # 1. terminal
    uvicorn backend.app.main:app --port 8000
    # 2. terminal
    python -m benchmarks.bench_concurrent
    python -m benchmarks.bench_concurrent --base-url http://127.0.0.1:8000 --user-id 1
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time

from benchmarks._harness import Measurement, print_table, save, skipped

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
QUESTIONS = [
    "Turing makinesi nedir?",
    "Context free grammar nedir?",
    "Push down automata nasıl çalışır?",
    "Sonlu durum makinesi ile Turing makinesi farkı nedir?",
    "Bağlamdan bağımsız dilbilgisi örneği ver",
]


def _server_alive(base_url: str) -> bool:
    import httpx

    try:
        resp = httpx.get(f"{base_url}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _one_ask(base_url: str, user_id: int, question: str, timeout: float) -> dict:
    """Tek bir `/ask/stream` isteği; TTFT ve toplam süreyi döner."""
    import httpx

    start = time.perf_counter()
    first_token_at = None
    tokens = 0
    error = None
    try:
        with httpx.stream(
            "POST",
            f"{base_url}/ask/stream",
            json={"question": question, "document_ids": None, "language": "tr"},
            headers={"X-User-Id": str(user_id)},
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}", "total_ms": (time.perf_counter() - start) * 1000}
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[len("data:"):].strip())
                if event.get("type") == "token":
                    if first_token_at is None:
                        first_token_at = time.perf_counter() - start
                    tokens += 1
                elif event.get("type") == "error":
                    error = event.get("detail")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    return {
        "ttft_ms": first_token_at * 1000 if first_token_at is not None else None,
        "total_ms": (time.perf_counter() - start) * 1000,
        "tokens": tokens,
        "error": error,
    }


def _run_wave(base_url: str, user_id: int, n: int, timeout: float) -> list[dict]:
    """N isteği AYNI ANDA başlatır (thread başına bir istek)."""
    results: list[dict] = [None] * n  # type: ignore[list-item]
    barrier = threading.Barrier(n)

    def worker(i: int):
        barrier.wait()  # hepsi gerçekten aynı anda başlasın
        results[i] = _one_ask(base_url, user_id, QUESTIONS[i % len(QUESTIONS)], timeout)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def _health_latency_under_load(base_url: str, n_background: int, timeout: float) -> Measurement:
    """Arka planda N soru akarken `/health`'in gecikmesi.

    Sağlıklı bir sunucuda `/health` her zaman milisaniyeler sürmeli. Uzuyorsa
    event loop veya threadpool tıkanmış demektir (audit P0-3, P1-6).
    """
    import httpx

    m = Measurement(
        name=f"health_latency_{n_background}_akis_altinda",
        note="event loop/threadpool tıkanıklığı göstergesi (audit P0-3, P1-6)",
    )
    stop = threading.Event()

    def noise():
        while not stop.is_set():
            _one_ask(base_url, 1, QUESTIONS[0], timeout)

    threads = [threading.Thread(target=noise, daemon=True) for _ in range(n_background)]
    for t in threads:
        t.start()
    time.sleep(2)  # yük otursun
    try:
        for _ in range(20):
            start = time.perf_counter()
            try:
                httpx.get(f"{base_url}/health", timeout=timeout)
                m.samples.append((time.perf_counter() - start) * 1000)
            except Exception as exc:
                m.skipped_reason = f"{type(exc).__name__}: {exc}"
                break
            time.sleep(0.25)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=1)
    return m


def run(base_url: str, user_id: int, levels: list[int], timeout: float):
    measurements: list[Measurement] = []
    extra: dict = {"base_url": base_url, "user_id": user_id, "levels": levels, "waves": {}}

    if not _server_alive(base_url):
        return [skipped("tumu", f"{base_url} adresinde çalışan bir backend yok")], extra

    # Baseline `/health` (yüksüz)
    idle = Measurement(name="health_latency_yuksuz", note="referans değer")
    import httpx

    for _ in range(20):
        start = time.perf_counter()
        httpx.get(f"{base_url}/health", timeout=timeout)
        idle.samples.append((time.perf_counter() - start) * 1000)
    measurements.append(idle)

    for n in levels:
        results = _run_wave(base_url, user_id, n, timeout)
        extra["waves"][str(n)] = results

        ttfts = [r["ttft_ms"] for r in results if r.get("ttft_ms") is not None]
        totals = [r["total_ms"] for r in results if r.get("total_ms") is not None]
        errors = [r["error"] for r in results if r.get("error")]

        measurements.append(
            Measurement(
                name=f"ttft_{n}_esszamanli",
                samples=ttfts,
                note=f"{n} eşzamanlı /ask/stream · hata: {len(errors)}/{n}",
                metadata={"errors": errors},
            )
        )
        measurements.append(
            Measurement(
                name=f"toplam_{n}_esszamanli",
                samples=totals,
                note=f"{n} eşzamanlı /ask/stream",
            )
        )
        if errors:
            print(f"  ! {n} eşzamanlıda {len(errors)} hata: {errors[:3]}")
        # GPU'nun toparlanması için istekler arası nefes payı.
        time.sleep(2)

    measurements.append(_health_latency_under_load(base_url, 3, timeout))

    # Hata oranı özeti
    extra["error_rate"] = {
        str(n): sum(1 for r in extra["waves"][str(n)] if r.get("error")) / max(n, 1)
        for n in levels
    }
    return measurements, extra


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--levels", default="1,2,5,10",
                        help="Ölçülecek eşzamanlılık seviyeleri (virgülle)")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    measurements, extra = run(args.base_url, args.user_id, levels, args.timeout)
    print_table("concurrent", measurements)

    if extra.get("error_rate"):
        print("\n  Hata oranı:")
        for n, rate in extra["error_rate"].items():
            print(f"    {n:>3} eşzamanlı: %{rate * 100:.0f}")

    path = save("concurrent", measurements, extra)
    print(f"\n  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
