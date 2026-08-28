"""LLM (Foundry Local) benchmark'ı: warmup, time-to-first-token, üretim hızı.

NE ÖLÇÜYOR
----------
- `model_warmup_soguk`   : `foundry server start` + `foundry model load` süresi
                           (backend açılışında bir kez ödeniyor, bkz. main.py:_warmup)
- `ttft_*`               : ilk GÖRÜNÜR token'a kadar geçen süre — kullanıcının
                           "cevap gelmeye başladı" olarak algıladığı an
- `toplam_uretim_*`      : akışın tamamı
- `token_per_second`     : üretim hızı (uzunluktan bağımsız karşılaştırma için)

ÖNEMLİ — `_stream_with_warmup` ETKİSİ
------------------------------------
`llm.py:_stream_with_warmup` ilk parçayı, en az 4 görünür karakter birikene
kadar tamponluyor. Yani buradaki TTFT, modelin ilk token'ını değil,
KULLANICININ GÖRDÜĞÜ ilk metni ölçüyor — doğru metrik bu.

ÇALIŞTIRMA ŞARTI
----------------
Foundry Local servisinin ayakta ve modelin yüklü olması gerekiyor. Değilse
benchmark çöker değil, ATLANIR.

    python -m benchmarks.bench_llm
"""
from __future__ import annotations

import sys
import time

from benchmarks._harness import (
    Measurement,
    gpu_used_mb,
    print_table,
    process_rss_mb,
    save,
    skipped,
    timed,
)

# Ölçüm prompt'ları. Context uzunluğu prefill süresini doğrudan belirlediği
# için üç ayrı senaryo: bağlamsız, kısa bağlam, dolu bağlam (üretimdeki
# max_context_chars sınırına yakın).
_SHORT_CONTEXT = ["Turing makinesi, sonsuz bir bant üzerinde çalışan soyut bir hesaplama modelidir."]
_FULL_CONTEXT = [
    (
        "Turing makinesi, sonsuz uzunlukta bir bant, bant üzerinde hareket eden bir "
        "okuma-yazma kafası ve sonlu bir durum kümesinden oluşan soyut bir hesaplama "
        "modelidir. Her adımda makine, bulunduğu durumu ve kafanın okuduğu sembolü "
        "kullanarak bir geçiş fonksiyonuna başvurur. "
    )
    * 12
] * 5


def _iter_stream_timed(gen):
    """Bir generator'ı tüketirken ilk parçanın ve toplamın süresini ölçer."""
    start = time.perf_counter()
    first_at = None
    chars = 0
    pieces = 0
    for piece in gen:
        if first_at is None:
            first_at = time.perf_counter() - start
        chars += len(piece)
        pieces += 1
    total = time.perf_counter() - start
    return {
        "ttft_ms": (first_at * 1000.0) if first_at is not None else None,
        "total_ms": total * 1000.0,
        "chars": chars,
        "pieces": pieces,
        # Kaba token tahmini: Türkçe/İngilizce karışık metinde ~4 karakter/token.
        "approx_tokens_per_second": (chars / 4) / total if total > 0 else None,
    }


def run():
    from backend.app.core.config import settings
    from backend.app.services import llm

    measurements: list[Measurement] = []
    extra: dict = {
        "model_alias": settings.foundry_model_alias,
        "model_has_thinking": settings.model_has_thinking,
        "answer_token_budgets": settings.answer_token_budgets,
        "vram_used_mb_before": gpu_used_mb(),
    }

    # --- 1) Soğuk warmup --------------------------------------------------
    # Kasıtlı olarak `warmup=0`: ölçmek istediğimiz şey tam da soğuk başlangıç.
    # `_get_manager`/`_get_model_id` lru_cache'li olduğu için bu YALNIZCA bu
    # süreçte ilk kez çalıştırıldığında gerçek soğuk süreyi verir.
    warm = timed(
        "model_warmup_soguk",
        llm.warmup,
        repeat=1,
        warmup=0,
        note="main.py:_warmup — backend açılışında bir kez; ilk sorunun gizlenen maliyeti",
    )
    measurements.append(warm)

    try:
        llm._get_manager()
        llm._get_model_id()
    except Exception as exc:
        reason = f"Foundry Local kullanılamıyor: {type(exc).__name__}: {exc}"
        for name in ("ttft_baglamsiz", "ttft_kisa_baglam", "ttft_dolu_baglam",
                     "toplam_uretim_baglamsiz", "toplam_uretim_kisa_baglam",
                     "toplam_uretim_dolu_baglam"):
            measurements.append(skipped(name, reason))
        extra["error"] = reason
        return measurements, extra

    extra["vram_used_mb_after_load"] = gpu_used_mb()
    extra["rss_mb_after_load"] = process_rss_mb()

    # --- 2) TTFT ve toplam üretim ----------------------------------------
    scenarios = [
        ("baglamsiz", "Turing makinesi nedir?", [], False),
        ("kisa_baglam", "Turing makinesi nedir?", _SHORT_CONTEXT, True),
        ("dolu_baglam", "Turing makinesi nasıl çalışır?", _FULL_CONTEXT, True),
    ]

    extra["runs"] = {}
    for label, question, context, has_context in scenarios:
        ttft = Measurement(
            name=f"ttft_{label}",
            note=f"llm.py:generate_answer_stream — ilk GÖRÜNÜR metin · context {sum(len(c) for c in context)} karakter",
        )
        total = Measurement(
            name=f"toplam_uretim_{label}",
            note="akışın tamamı (retrieval hariç)",
        )
        runs = []
        # 3 tekrar: yerel modellerin örnekleme varyansı yüksek, tek ölçüm
        # yanıltıcı olurdu.
        for _ in range(3):
            try:
                stats = _iter_stream_timed(
                    llm.generate_answer_stream(question, context, "tr", has_context)
                )
            except Exception as exc:
                ttft.skipped_reason = f"{type(exc).__name__}: {exc}"
                total.skipped_reason = ttft.skipped_reason
                break
            runs.append(stats)
            if stats["ttft_ms"] is not None:
                ttft.samples.append(stats["ttft_ms"])
            total.samples.append(stats["total_ms"])
        extra["runs"][label] = runs
        measurements.append(ttft)
        measurements.append(total)

    # --- 3) Üretim hızı ---------------------------------------------------
    tps = [
        r["approx_tokens_per_second"]
        for runs in extra["runs"].values()
        for r in runs
        if r.get("approx_tokens_per_second")
    ]
    if tps:
        m = Measurement(name="token_per_second", unit="tok/s", samples=tps,
                        note="~4 karakter/token varsayımıyla kaba tahmin")
        measurements.append(m)

    extra["vram_used_mb_after_generation"] = gpu_used_mb()
    return measurements, extra


def main() -> int:
    measurements, extra = run()
    print_table("llm", measurements)
    if extra.get("error"):
        print(f"\n  NOT: {extra['error']}")
        print("  Foundry Local'i başlatıp tekrar deneyin: foundry server start")
    path = save("llm", measurements, extra)
    print(f"\n  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
