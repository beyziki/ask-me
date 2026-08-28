"""Örnekleme (sampling) ayarlarının tekrar döngüsüne etkisini ölçer.

BULGULAR (2026-08-27, ministral-3-3b-instruct-2512, CUDA GPU)
------------------------------------------------------------
Bu araçla ölçülen ve İKİ HİPOTEZİ DE DÜZELTEN sonuçlar:

1. `frequency_penalty` / `presence_penalty` GERÇEKTEN ÇALIŞIYOR.
   Aynı `random_seed` ile penalty 0.0 ve 2.0 farklı metin üretti. (Önce
   bunların ONNX Runtime GenAI'ye ulaşmadığı sanılıyordu; yanlışmış.)

2. `temperature`, `top_p`, `top_k` YOK SAYILIYOR.
   temperature 0.2 / 0.7 / 1.0, top_p 0.9 / 0.95 ve top_k 40 — beş ayrı
   config BİREBİR AYNI metni üretti (aynı karakter sayısı, aynı hash).
   Yani Foundry Local altındaki motor GREEDY çözümleme yapıyor: ORT
   GenAI'nin `do_sample` varsayılanı `false` ve o haldeyken temperature /
   top_p / top_k'nın hiçbir anlamı yok.

   BUNUN SONUCU: üretim DETERMİNİSTİK. Aynı prompt her zaman aynı cevabı
   veriyor. Tekrar döngüsü de "bazen olan" bir şey değil — belirli
   prompt'larda HER ZAMAN oluyor, diğerlerinde hiç olmuyor.

DENEY TASARIMI — NEDEN "AYNI SORUYU N KEZ" DEĞİL
------------------------------------------------
Aracın ilk hali her config için aynı soruyu 5 kez çalıştırıyordu. Üretim
deterministik olduğu için bu 5 kez aynı cevabı almak demekti; ölçüm hiçbir
şey öğretmiyordu (altı config de 0/5).

Doğrusu: SORU KÜMESİNİ değiştirmek. Her config, her soruyu BİR kez
çalıştırıyor; döngü oranı sorular üzerinden hesaplanıyor. Böylece hem
maliyet düşüyor hem de "hangi prompt döngüye sokuyor" sorusunun cevabı
çıkıyor.

ÖLÇÜLEN SORUN
-------------
Gözlemlenen arıza (bkz. docs/foundry-local-notlari.md §2): model bozuk bir
tekrar döngüsüne giriyor —

    "... (örn. sıvı sıcaklığının, sıvı sıvı sıvı sıvı sıvı sıvı sıvı ..."

Bu, context DOĞRU bulunduğunda bile oluyor, yani retrieval sorunu değil.
Greedy çözümlemede tekrar döngüsü kendi kendini besleyen bir çekim
noktasıdır: model bir kez "sıvı sıvı" ürettiğinde en yüksek olasılıklı
sonraki token yine "sıvı" olur ve çıkış yolu kalmaz.

KULLANIM
--------
    python -m evals.sampling_probe --determinism   # parametreler çalışıyor mu
    python -m evals.sampling_probe                 # config x soru taraması
    python -m evals.sampling_probe --only "fp 1.2" # tek config

ÖN KOŞUL: Foundry Local çalışıyor ve model yüklü olmalı
(`foundry model load ...`). Üretim veritabanının GEÇİCİ KOPYASI kullanılır.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# `settings` import zamanında bir kez kuruluyor (bkz. run_eval.py'deki aynı not).
_TMP = Path(tempfile.mkdtemp(prefix="ask-me-sampling-"))
_REAL_DB = PROJECT_ROOT / "data" / "ask_me.db"
if _REAL_DB.exists():
    shutil.copy2(_REAL_DB, _TMP / "probe.db")
    os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'probe.db'}"
os.environ.setdefault("VECTORSTORE_DIR", str(PROJECT_ROOT / "data" / "vectorstore"))
os.environ["WARMUP_ON_STARTUP"] = "false"
# Bu araç önce embedding modelini (prompt kurmak için), sonra Foundry
# Local'in dil modelini kullanıyor. İkisi aynı GPU'ya sığmıyor: prompt
# önbelleği boşken embedder GPU'ya yerleşip Foundry'yi "CudaMallocArray -
# out of memory" ile düşürdü. Embedder burada zaten yalnızca birkaç kısa
# soruyu gömüyor, CPU fazlasıyla yeterli.
os.environ["EMBEDDING_DEVICE"] = "cpu"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PROMPT_CACHE = RESULTS_DIR / "_probe_prompts.json"
QUESTIONS_PATH = Path(__file__).resolve().parent / "probe_questions.json"

DEFAULT_OWNER = 1


# Döngüye girme riski en yüksek soru tipi LİSTELEME isteyenler: model
# sayacak madde bittiğinde son maddeyi tekrarlamaya başlıyor (gözlemlenen
# arıza tam olarak bir "örn. ..." listesinin ortasında başladı). Bu yüzden
# küme bilinçli olarak açıklama + listeleme sorularından oluşuyor.
DEFAULT_QUESTIONS = [
    {"q": "IoT Tabanlı Tramvay İzleme ve Kestirimci Bakım Platformunu açıklar mısın?", "doc": 32},
    {"q": "Hangi teknolojiler ve araçlar kullanılmış, hepsini say", "doc": 32},
    {"q": "BabyRobot projesinin bileşenlerini tek tek açıkla", "doc": 30},
    {"q": "BabyRobot nasıl çalışıyor?", "doc": 30},
    {"q": "Turing makinesi nedir?", "doc": 16},
    {"q": "Bu bölümdeki ana konuları listele", "doc": 2},
]


# Denenecek ayarlar.
#
# MEZAR TAŞI: burada `temperature` / `top_p` / `top_k` taraması YOK, çünkü
# ölçüldü ve üçünün de bu kurulumda HİÇBİR ETKİSİ OLMADIĞI görüldü (bkz.
# dosya başındaki bulgular). Tekrar eklemeden önce `--determinism` benzeri
# bir kontrolle etkili olduklarını doğrula; aksi halde saatlerce aynı metni
# ölçersin. Geriye kalan tek gerçek kaldıraç penalty'ler.
CONFIGS = [
    {
        "name": "mevcut (fp .4 / pp .2)",
        "params": {"temperature": 0.2, "frequency_penalty": 0.4, "presence_penalty": 0.2},
        "note": "kodun bugünkü hali",
    },
    {
        "name": "penalty YOK (kontrol)",
        "params": {"temperature": 0.2},
        "note": "referans çizgisi: penalty olmadan ne oluyor",
    },
    {
        "name": "fp .8 / pp .4",
        "params": {"temperature": 0.2, "frequency_penalty": 0.8, "presence_penalty": 0.4},
        "note": "mevcut değerin iki katı",
    },
    {
        "name": "fp 1.2 / pp .6",
        "params": {"temperature": 0.2, "frequency_penalty": 1.2, "presence_penalty": 0.6},
        "note": "agresif",
    },
    {
        "name": "fp 1.6 / pp .8",
        "params": {"temperature": 0.2, "frequency_penalty": 1.6, "presence_penalty": 0.8},
        "note": "üst sınır — Türkçe akıcılığı bozuluyor mu?",
    },
]


# --- Parametre yönlendirme -----------------------------------------------
# OpenAI Python SDK'sinin `chat.completions.create` imzası TİPLENMİŞ: orada
# olmayan bir anahtar kelime argümanı sunucuya hiç gitmeden `TypeError` ile
# reddediliyor. Foundry Local'in desteklediği ama OpenAI şemasında olmayan
# parametreler (`top_k`, `random_seed`, `ep`, `ttl`) bu yüzden `extra_body`
# içinde gönderilmek zorunda.
#
# ÖNEMLİ: aynı kısıt ÜRETİM KODU için de geçerli. `services/llm.py`'deki
# `_create_chat_completion(**kwargs)` doğrudan SDK'ya geçirdiği için, oraya
# `top_k` eklemek istersen `extra_body={"top_k": ...}` şeklinde olmalı.
_SDK_NATIVE_PARAMS = {
    "temperature", "top_p", "frequency_penalty", "presence_penalty",
    "max_tokens", "max_completion_tokens", "stop", "seed", "n",
    "logit_bias", "user",
}


def _split_params(params: dict) -> tuple[dict, dict]:
    """(sdk_kwargs, extra_body) — bkz. `_SDK_NATIVE_PARAMS`."""
    native = {k: v for k, v in params.items() if k in _SDK_NATIVE_PARAMS}
    extra = {k: v for k, v in params.items() if k not in _SDK_NATIVE_PARAMS}
    return native, extra


# --- Tekrar ölçütleri -----------------------------------------------------
def _normalize(word: str) -> str:
    """`llm.py:_normalize_word` ile aynı: büyük/küçük harf ve noktalama
    farkını yok sayar (döngüdeki öbek her turda birebir aynı yazılmıyor)."""
    return "".join(ch for ch in word.lower() if ch.isalnum())


def _repeat_score(text: str) -> float:
    """0..1 arası "tekrarlılık": 1 - (benzersiz kelime / kelime)."""
    words = [w for w in text.lower().split() if w]
    if not words:
        return 0.0
    return 1.0 - len(set(words)) / len(words)


def _repetition_pressure(text: str) -> int:
    """Metinde ARDIŞIK olarak en çok kaç kez tekrarlanan bir blok var?

    NEDEN GEREKLİ: "döngüye girdi mi?" ikili bir ölçüt; hiçbir çalıştırma
    guard'ı tetiklemezse tablo hiçbir şeyi ayırt etmiyor. Bu ölçüt sürekli:
    1 = hiç ardışık tekrar yok. Guard 2–12 kelimelik bloğun 4 kez
    tekrarında tetikleniyor, yani 3 "kıl payı kaçtı" demek — asıl bakılacak
    erken uyarı bu.
    """
    words = [w for w in (_normalize(x) for x in text.split()) if w]
    n = len(words)
    best = 1
    for i in range(n):
        for period in range(1, 13):
            if i + 2 * period > n:
                break
            block = words[i : i + period]
            reps, j = 1, i + period
            while j + period <= n and words[j : j + period] == block:
                reps += 1
                j += period
            if reps > best:
                best = reps
    return best


# --- Model çağrısı --------------------------------------------------------
def _client_and_model():
    from openai import OpenAI

    from backend.app.services.llm import _get_manager, _get_model_id

    manager = _get_manager()
    return OpenAI(base_url=manager.endpoint, api_key=manager.api_key), _get_model_id()


def _run_once(client, model_id, messages, params, max_tokens):
    from backend.app.services.llm import _looks_degenerate, strip_think

    native, extra = _split_params(params)
    kwargs = {"model": model_id, "messages": messages, "max_tokens": max_tokens, **native}
    if extra:
        kwargs["extra_body"] = extra

    start = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    text = strip_think(response.choices[0].message.content or "")
    return {
        "text": text,
        "chars": len(text),
        "degenerate": _looks_degenerate(text),
        "pressure": _repetition_pressure(text),
        "repeat_score": _repeat_score(text),
        "latency_ms": elapsed,
        "finish_reason": getattr(response.choices[0], "finish_reason", None),
    }


# --- Prompt hazırlama -----------------------------------------------------
def _retrieve_messages(question: str, owner_id: int, doc_id: int | None):
    """Gerçek retrieval hattından geçerek prompt'u kurar.

    PAHALI: `_retrieve` embedding modelini (sentence-transformers + torch)
    yüklüyor. Bu yüzden sonuç `PROMPT_CACHE`'e yazılıyor; sonraki
    çalıştırmalar modeli bir daha hiç yüklemiyor.
    """
    try:
        from backend.app.api.ask import _retrieve
        from backend.app.db.base import SessionLocal
        from backend.app.models.schemas import AskRequest
        from backend.app.services.llm import build_prompt
    except ImportError as exc:
        print("HATA: proje bağımlılıkları yüklenemedi:")
        print(f"  {type(exc).__name__}: {exc}")
        print(f"\n  Kullanılan Python: {sys.executable}")
        print("\n  Sanal ortam aktif mi? Proje kökünde:")
        print("      .venv\\Scripts\\activate")
        raise SystemExit(2) from exc

    payload = AskRequest(
        question=question,
        document_ids=[doc_id] if doc_id else None,
        language="tr",
    )
    db = SessionLocal()
    try:
        context_chunks, _sources, language, has_context = _retrieve(payload, owner_id, db)
    finally:
        db.close()
    return build_prompt(question, context_chunks, language, has_context), context_chunks, has_context


def _load_questions() -> list[dict]:
    if QUESTIONS_PATH.exists():
        return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return DEFAULT_QUESTIONS


def _build_all_prompts(questions: list[dict], owner_id: int, rebuild: bool = False) -> list[dict]:
    """Her soru için prompt'u önbellekten okur, yoksa retrieval'ı çalıştırır.

    Önbellek anahtarı (soru, owner, doc) üçlüsü — biri değişirse o girdi
    kendiliğinden geçersizleşiyor, bayat prompt üzerinde ölçüm riski yok.
    """
    cache = {}
    if not rebuild and PROMPT_CACHE.exists():
        try:
            cache = json.loads(PROMPT_CACHE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}

    prepared, missing = [], 0
    for item in questions:
        key = f"{owner_id}|{item.get('doc')}|{item['q']}"
        if key in cache:
            entry = cache[key]
        else:
            if missing == 0:
                print("prompt    : retrieval çalıştırılıyor (embedding modeli yükleniyor)...")
            missing += 1
            messages, chunks, has_context = _retrieve_messages(item["q"], owner_id, item.get("doc"))
            entry = {"messages": messages, "n_chunks": len(chunks), "has_context": has_context}
            cache[key] = entry
        prepared.append({**item, **entry})

    if missing:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        PROMPT_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(f"prompt    : {len(prepared)} soru önbellekten (--rebuild-prompt ile yenile)")
    return prepared


# --- Determinizm probe'u --------------------------------------------------
def run_determinism_probe(messages, max_tokens: int, repeat: int = 3) -> dict:
    """Hangi parametreler GERÇEKTEN sampler'a ulaşıyor?

    Önce `random_seed` onurlandırılıyor mu diye bakıyoruz (aynı seed + aynı
    parametrelerle iki çağrı). Onurlandırılıyorsa her parametreyi tek tek
    değiştirip çıktının değişip değişmediğine bakabiliyoruz: değişmiyorsa o
    parametre yok sayılıyor demektir.
    """
    client, model_id = _client_and_model()
    base = {"temperature": 0.2, "random_seed": 1234}

    a = _run_once(client, model_id, messages, base, max_tokens)
    b = _run_once(client, model_id, messages, base, max_tokens)
    seed_works = a["text"] == b["text"]
    print(f"  aynı ayarla iki çağrı aynı mı : {'EVET (üretim deterministik)' if seed_works else 'HAYIR'}")

    if not seed_works:
        return {
            "deterministic": False,
            "verdict": (
                "Üretim deterministik DEĞİL — parametreleri tek tek izole etmek için "
                "istatistiksel karşılaştırma gerekir. (Beklenmedik: greedy çözümlemede "
                "deterministik olmalıydı.)"
            ),
        }

    # Deterministik: her parametreyi tek tek değiştirip etkisini izole et.
    probes = {
        "frequency_penalty": {**base, "frequency_penalty": 2.0},
        "presence_penalty": {**base, "presence_penalty": 2.0},
        "temperature": {**base, "temperature": 1.5},
        "top_p": {**base, "top_p": 0.5},
        "top_k": {**base, "top_k": 5},
    }
    results = {}
    for name, params in probes.items():
        try:
            out = _run_once(client, model_id, messages, params, max_tokens)
        except Exception as exc:
            results[name] = {"effective": None, "error": f"{type(exc).__name__}: {exc}"}
            print(f"  {name:<20} HATA: {type(exc).__name__}")
            continue
        effective = out["text"] != a["text"]
        results[name] = {"effective": effective}
        print(f"  {name:<20} {'ETKİLİ' if effective else 'YOK SAYILIYOR'}")

    dead = [k for k, v in results.items() if v.get("effective") is False]
    return {
        "deterministic": True,
        "parameters": results,
        "verdict": (
            f"Yok sayılan parametreler: {', '.join(dead)}. Bunları ayarlamanın hiçbir "
            f"etkisi yok — üretim greedy çözümleme yapıyor."
            if dead
            else "Tüm parametreler etkili."
        ),
    }


# --- Ana tarama -----------------------------------------------------------
def run_sweep(prompts: list[dict], configs: list[dict], max_tokens: int) -> list[dict]:
    client, model_id = _client_and_model()
    rows = []
    for config in configs:
        runs = []
        for item in prompts:
            try:
                out = _run_once(client, model_id, item["messages"], config["params"], max_tokens)
            except Exception as exc:
                print(f"  ! {config['name']}: {type(exc).__name__}: {exc}")
                runs = []
                break
            out["question"] = item["q"]
            runs.append(out)
            flag = "DÖNGÜ" if out["degenerate"] else f"baskı {out['pressure']}"
            print(f"    {config['name']:<24} {item['q'][:38]:<40} {flag}")
        if not runs:
            rows.append({**config, "skipped": True})
            continue
        loops = sum(1 for r in runs if r["degenerate"])
        rows.append({
            "name": config["name"],
            "note": config["note"],
            "params": config["params"],
            "questions": len(runs),
            "loop_count": loops,
            "loop_rate_pct": round(100 * loops / len(runs), 1),
            "max_pressure": max(r["pressure"] for r in runs),
            "mean_pressure": round(sum(r["pressure"] for r in runs) / len(runs), 2),
            "mean_repeat_score": round(sum(r["repeat_score"] for r in runs) / len(runs), 4),
            "mean_chars": round(sum(r["chars"] for r in runs) / len(runs)),
            "hit_token_cap": sum(1 for r in runs if r["finish_reason"] == "length"),
            "mean_latency_ms": round(sum(r["latency_ms"] for r in runs) / len(runs)),
            "per_question": [
                {"q": r["question"], "pressure": r["pressure"], "chars": r["chars"],
                 "degenerate": r["degenerate"], "finish_reason": r["finish_reason"]}
                for r in runs
            ],
            "sample": runs[0]["text"][:400],
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Foundry Local örnekleme taraması")
    parser.add_argument("--owner", type=int, default=DEFAULT_OWNER)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--determinism", action="store_true",
                        help="hangi parametreler gerçekten etkili, onu ölç")
    parser.add_argument("--only", default=None, help="yalnızca adı bunu içeren config")
    parser.add_argument("--rebuild-prompt", action="store_true")
    parser.add_argument("--label", default="sampling")
    args = parser.parse_args()

    questions = _load_questions()
    prompts = _build_all_prompts(questions, args.owner, rebuild=args.rebuild_prompt)
    no_ctx = [p["q"] for p in prompts if not p["has_context"]]
    print(f"soru      : {len(prompts)} adet")
    if no_ctx:
        print(f"  UYARI: {len(no_ctx)} soruda context bulunamadı — bunlar 'context varken")
        print(f"         döngü' senaryosunu temsil etmiyor: {no_ctx}")
    print()

    if args.determinism:
        print("--- hangi parametreler gerçekten etkili? ---")
        verdict = run_determinism_probe(prompts[0]["messages"], args.max_tokens)
        print(f"\n  SONUÇ: {verdict['verdict']}")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "sampling_determinism.json"
        out.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  -> {out}")
        return 0

    configs = CONFIGS
    if args.only:
        configs = [c for c in CONFIGS if args.only.lower() in c["name"].lower()]
        if not configs:
            print(f"'{args.only}' ile eşleşen config yok.")
            return 1

    print(f"tarama    : {len(configs)} config x {len(prompts)} soru = "
          f"{len(configs) * len(prompts)} üretim\n")
    rows = run_sweep(prompts, configs, args.max_tokens)

    print("\n" + "=" * 96)
    print(f"  {'config':<24} {'döngü':>7} {'baskı ort/max':>14} {'tekrar':>8} "
          f"{'karakter':>9} {'tavan':>7} {'süre':>9}")
    print("-" * 96)
    for row in rows:
        if row.get("skipped"):
            print(f"  {row['name']:<24} {'ATLANDI':>30}")
            continue
        print(f"  {row['name']:<24} {row['loop_count']:>3}/{row['questions']:<3} "
              f"{row['mean_pressure']:>9}/{row['max_pressure']:<3} "
              f"{row['mean_repeat_score']:>8.3f} {row['mean_chars']:>9} "
              f"{row['hit_token_cap']:>4}/{row['questions']:<2} "
              f"{row['mean_latency_ms']:>7} ms")
    print("=" * 96)
    print("""
  NASIL OKUNUR
  ------------
  döngü   : guard'ı tetikleyen SORU sayısı (üretim deterministik olduğu için
            aynı soruyu tekrarlamanın anlamı yok — küme sorulardan oluşuyor).
  baskı   : en uzun ardışık blok tekrarı (ort/max). 1 = temiz, guard 4'te
            tetikleniyor, 3 = kıl payı kaçtı. ASIL ERKEN UYARI BU.
  tekrar  : 1 - (benzersiz kelime / kelime).
  tavan   : kaç cevap `max_tokens` sınırına çarparak kesildi. Hepsi
            çarpıyorsa model doğal olarak bitmiyor, dolduruyor demektir —
            döngüler tam orada başlıyor.

  Kazanan: döngü ve baskı en düşük, karakter sayısı hâlâ makul olan.
  Penalty'yi aşırı yükseltmek Türkçe akıcılığını bozar; `sample` alanını
  gözle oku.""")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{args.label}.json"
    path.write_text(
        json.dumps({"questions": [p["q"] for p in prompts], "configs": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
