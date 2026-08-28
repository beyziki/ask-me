# PHASE 1 — Ölçüm ve Benchmark Altyapısı

**Tarih:** 2026-08-17
**Kural:** Bu fazda **hiçbir üretim kodu değiştirilmedi.** `backend/` ve
`frontend-web/src/` altındaki tek bir satır bile aynı.

---

## 1. Hangi dosyalar değişti

### Yeni

| Dosya | Ne |
|---|---|
| `pyproject.toml` | pytest yapılandırması: `testpaths`, markerlar, uyarı filtreleri |
| `tests/conftest.py` | **Veritabanı izolasyonu** + embedding-bağımlı testleri otomatik atlama + ortak fixture'lar |
| `benchmarks/_harness.py` | Ortak altyapı: zamanlama, medyan/p95, ortam bilgisi, JSON çıktı |
| `benchmarks/bench_retrieval.py` | DB, BM25, FAISS, sorgu embedding'i |
| `benchmarks/bench_ingestion.py` | PDF parse, chunking, embedding, FAISS yazma |
| `benchmarks/bench_llm.py` | Warmup, time-to-first-token, üretim, token/s |
| `benchmarks/bench_concurrent.py` | 1/2/5/10 eşzamanlı `/ask/stream`, yük altında `/health` |
| `benchmarks/bench_resources.py` | Kademeli RAM, istek başına RAM, önbellek büyümesi, VRAM |
| `benchmarks/bench_bundle.py` | Frontend chunk sayısı ve gzip boyutları |
| `benchmarks/run_all.py` | Hepsini **ayrı süreçlerde** koşar, özet üretir |
| `benchmarks/compare.py` | Önce/sonra tablosu; regresyonda çıkış kodu 1 |
| `benchmarks/README.md` | Kullanım + metodoloji |
| `evals/dataset.json` | **40 vakalık RAG doğruluk seti** (taslak — doğrulaman gerekiyor) |
| `evals/run_eval.py` | recall/precision/MRR/terim/has_context ölçümü |
| `evals/compare.py` | Vaka bazlı önce/sonra; bozulan varsa çıkış kodu 1 |
| `evals/README.md` | Kullanım + ground truth tasarım gerekçesi |
| `docs/PHASE1_REPORT.md` | Bu dosya |

### Değişen

| Dosya | Ne | Neden |
|---|---|---|
| `tests/test_quiz_utils.py` | Kırık testi düzelttim, sihirli sayıyı sabitle değiştirdim, 1 gözlem testi ekledim | Test **zaten kırmızıydı** (aşağıda) |
| `docs/ARCHITECTURE_PERFORMANCE_AUDIT.md` | 2 yeni P0 eklendi, P1-5 düzeltildi | Ölçüm yeni bulgu verdi ve bir iddiamı çürüttü |

**Üretim kodunda değişiklik: yok.**

---

## 2. Test durumu — önce ve sonra

| | Önce | Sonra |
|---|---|---|
| Geçen | 117 | **119** |
| Kalan | **7** | **0** |
| Atlanan | 0 | 6 *(embedding modeli olmayan ortamda; senin makinende çalışır)* |

### Neden 7 test kalıyordu

**1 tanesi gerçek bir kırık test:** `test_quiz_max_tokens_scales_with_question_count_below_ceiling`

```
assert _quiz_max_tokens(5) < 1000
E   assert 1200 < 1000
```

`_QUIZ_MAX_TOKENS_CEILING` bir noktada 1000 → 1200 yükseltilmiş,
`test_quiz_max_tokens_never_exceeds_ceiling` güncellenmiş ama bu test
güncellenmemiş. Kod yorumu hâlâ "1000" diyor (`quiz.py:44`) — yani kod ile
yorum da uyuşmuyor.

Testi sihirli sayı yerine **sabitin kendisine** bakacak şekilde yazdım ki aynı
kayma tekrar olmasın. **Üretim değerine dokunmadım** (1200 olarak kaldı) —
Faz 1'in kuralı bu.

Yanında bir gözlem testi ekledim: `test_default_quiz_size_already_saturates_the_ceiling`.
Çünkü varsayılan 5 soruda `100 + 5×220 = 1200` = tam tavan. Yani "soru
sayısına göre ölçekle" optimizasyonu **varsayılan senaryoda hiçbir tasarruf
sağlamıyor**. Bu Faz 3'te ele alınacak; test o değişikliği görünür kılmak için
duruyor.

**6 tanesi ortam eksikliği:** gerçek embedding üretimi gerektiriyorlar ve bu
sandbox'ta HuggingFace bloklu (403). Senin makinende model önbellekte olduğu
için çalışacaklar. `conftest.py` artık bunu **hata değil atlama** olarak ele
alıyor (CI için de gerekli) ve model varsa `slow` işaretliyor.

---

## 3. Faz 1'de bulunan iki yeni P0

Bunlar statik analizde görünmüyordu; **testleri çalıştırınca** ve **eval verisi
hazırlarken** ortaya çıktı.

### P0-7 · `pytest` üretim veritabanına yazıyordu → **düzeltildi**

`data/ask_me.db`'deki 9 kullanıcıdan **6'sı test fixture'ı**:
`alice_docs`, `alice_ml`, `bob_docs`, `carol_en`, `carol_tr`, `dup_user`.
Gerçek olanlar: `byzerdem`, `beyza`, `ben`.

**Sebep:** `settings` modül import zamanında bir kez kuruluyor.
`test_ingestion.py`, `test_llm_utils.py`, `test_quiz_utils.py`,
`test_rag_merge.py` hiç env değişkeni set etmeden `backend.app...` import
ediyor. Bunlardan biri önce yüklenirse `settings` gerçek veritabanına bağlanıyor
ve sonra gelen modüllerin env yazması etkisiz kalıyor.

Sadece kirlilik değildi: `test_delete_document_removes_it_and_ungroups_related_quiz`
**doküman siliyordu.** DB'nin %86'sının boş alan olması muhtemelen kısmen bundan.

**Düzeltme:** `tests/conftest.py` env'i her test modülünden önce ayarlıyor
(pytest conftest'i her zaman önce yükler). Ayrıca oturum başında izolasyonu
**doğrulayan** bir fixture var — düzen ileride bozulursa testler sessizce
üretime yazmak yerine anında ve açık bir mesajla kırmızıya döner.

**Doğrulandı:** tam test koşusu öncesi/sonrası `ask_me.db` MD5'i aynı.

### P0-6 · `.pptx` binary olarak indekslenmiş — korpusun %29'u çöp

`Chapter_6.pptx` → **297 chunk mojibake**, owner 1 korpusunun **%29.3'ü**.
Chunk #0'ın içeriği: `PK` (ZIP imzası).

`detect_file_type` yalnızca `.pdf`, `.md` ve kod uzantılarını tanıyor; diğer
**her şey** `"text"` sayılıp `read_text(errors="ignore")` ile okunuyor:

```
.pptx -> text    .docx -> text    .xlsx -> text    .zip -> text    .png -> text
```

Bu 297 chunk hem FAISS index'inde hem BM25 korpusunda; her soruda rastgele
token eşleşmeleriyle yarışıyor. Kullanıcıya hiçbir hata gösterilmiyor.

**Faz 4'e eklendi.** Bu fazda düzeltilmedi (üretim kodu değişikliği gerektiriyor).

---

## 4. Bir iddiam çürütüldü: P1-5

Audit raporunda `_index_cache` için şunu yazmıştım:

> *"Anahtar `(mtime, size)` içerdiği için her yüklemede yeni bir girdi
> oluşuyor, eski `faiss.Index` bellekte kalıyor — sınırsız sızıntı."*

**Yanlıştı.** `bench_resources.py` ile ölçtüm: aynı dosyaya 10 kez yazıp 10 kez
okuduktan sonra önbellekte **1 girdi** var. Sebep: önbellek **anahtarı**
`str(path)` (`rag.py:76`); `(mtime, size)` yalnızca **değerin** bir parçası,
geçerlilik kontrolü için. Aynı yola tekrar yazmak girdiyi üzerine yazıyor.

Gerçek sorun daha küçük ama hâlâ var: önbelleğin **hiç üst sınırı yok**, yani
kullanıcı sayısıyla büyüyor. Ölçüm: 10 farklı kullanıcı → 11 girdi, **+28 MB**
(küçük index'lerle). Audit raporu düzeltildi.

---

## 5. Referans ölçümler

> **Bunlar senin baseline'ın DEĞİL.** Bir Linux sandbox'ta alındı; embedding
> modeli ve Foundry Local orada yoktu. Kendi makinende
> `python -m benchmarks.run_all --label baseline` çalıştır.

### Retrieval — en önemli tablo

| Ölçüm | Medyan | Not |
|---|---|---|
| DB: tüm chunk'ları çek (1014 satır / 4.46 MB) | 9.4 ms | `ask.py:121`, her soruda |
| **BM25 korpus tokenizasyonu** | **405.9 ms** | `rag.py:258` — **önbelleğe rağmen her soruda** |
| BM25 index kurulumu | 220.2 ms | önbellekli, sadece ilk soru |
| **`bm25_search` uçtan uca** | **421.9 ms** | üretimdeki gerçek yol |
| FAISS `read_index` | 0.27 ms | |
| FAISS `search` (1014 vektör, top_k=8) | *ölçülemedi* | embedding modeli yok |

**Okunuşu:** `bm25_search`'ün 422 ms'sinin **406 ms'si** gereksiz
tokenizasyon. FAISS tarafı (0.27 ms) tamamen ihmal edilebilir. LLM dışındaki
gecikmenin neredeyse tamamı tek bir satırda.

### Ölçekle büyüme

| Korpus | Tokenizasyon | BM25 kurulum |
|---|---|---|
| 1.014 chunk (bugün) | 379 ms | 200 ms |
| 10.140 chunk | **6.727 ms** | 2.560 ms |
| 50.700 chunk | **31.031 ms** | 12.807 ms |

### Bellek

| Aşama | RSS | Delta |
|---|---|---|
| Başlangıç | 16.7 MB | |
| Backend import sonrası | 812.8 MB | **+796 MB** (torch, faiss, sentence-transformers) |
| Korpus belleğe alındıktan sonra | 830.3 MB | +17 MB — **istek başına** |
| BM25 önbelleği sonrası | 914.9 MB | **+85 MB** (tek kullanıcı) |
| 10 kullanıcı index önbelleği | 942.6 MB | +28 MB, üst sınır yok |

5 eşzamanlı istekte tahmini RSS: **~900 MB**.

### Frontend bundle

| Ölçüm | Değer |
|---|---|
| **JS chunk sayısı** | **1** — kod bölme yok (P2-5) |
| Toplam JS (gzip) | 147.3 kB |
| Toplam CSS (gzip) | 6.4 kB |
| **İlk yükleme (gzip)** | **153.6 kB** |
| Build süresi | ~5.0 s |

Giriş ekranı bile `react-markdown` + `remark-gfm` dahil tüm uygulamayı
indiriyor.

### Ölçülemeyenler (senin makinende çalışacak)

`ingestion` (PDF yok), `llm` (Foundry yok), `concurrent` (backend ayakta değil),
ve tüm embedding ölçümleri.

---

## 6. Eval seti

**40 vaka**, gerçek doküman içeriğinden türetildi ve `expected_documents`
alanları içerik taramasıyla doğrulandı.

Üç grup:
- **25 normal soru** — tüm dosyalarda arama (TR/EN karışık, kısa/uzun, sayısal/kavramsal)
- **6 doküman-filtreli** — audit P0-1'i doğrudan ölçüyor; `f-net-only` kontrol grubu
- **3 bağlamsız** — `has_context=false` beklenen sorular

İki katman ayrı ölçülüyor: **ham sıralama** (`semantic_search` + `bm25_search` +
`rrf_merge`) ve **teslim edilen** (`_retrieve`, yani eşik + dedupe + kırpma
sonrası). Aradaki boşluk, sorunun aramada mı yoksa eşikte mi olduğunu söylüyor.

**Ground truth chunk id değil, doküman + terim.** Sebep: Faz 2'de yeniden
indeksleme tüm chunk id'lerini değiştirecek; chunk id'ye bağlı bir ground
truth o andan sonra kullanılamaz olurdu — yani tam da ölçmek istediğimiz
iyileştirmeyi ölçemezdik.

**Runner doğrulandı:** sahte bir embedder ile tüm kod yolu (metrikler, toplama,
rapor, JSON, `compare`) uçtan uca çalıştırıldı. Sahte embedder'la bile
P0-1 sinyali görünüyor: filtreli vakalarda ortalama semantic hit **1.7**,
filtresizlerde **8.0**.

---

## 7. Senden ihtiyacım olan iki şey

**1. Baseline'ı kendi makinende al.**

```bash
cd C:\Users\monster\Documents\Microsoft-Internship\ask-me

pytest                                          # 125 test yeşil olmalı
python -m benchmarks.run_all --label baseline   # Foundry açıksa llm de ölçülür
python -m evals.run_eval --label baseline

# concurrent için ayrı terminalde backend gerekiyor:
#   uvicorn backend.app.main:app --port 8000
# sonra:
python -m benchmarks.bench_concurrent --user-id 1
```

`benchmarks/results/baseline.json` ve `evals/results/baseline.json` oluşacak.
Faz 2'den itibaren her değişiklik bunlara karşı ölçülecek.

**2. Eval setini gözden geçir.**

`evals/dataset.json` bir **taslak**. Yanlış bir ground truth, ondan sonraki
her ölçümü sessizce bozar. Bakman gerekenler:
- Soru anlamlı mı?
- `expected_documents` doğru mu?
- `required_terms` gerçekten o dokümana özgü mü?
- Eklemek istediğin soru var mı?

Yanlış bulduklarını sil veya düzelt — dosya sade JSON, her vakanın `note`
alanında neyi ölçtüğü yazıyor.

---

## 8. Faz 2'de ne olacak

Sırayla, her biri ayrı ayrı ölçülerek:

1. **P0-2** BM25 token önbelleği — tek dosya, en yüksek kazanç/risk oranı (−406 ms/soru)
2. **P0-1** doküman filtreli semantic search — eval'deki `f-*` vakaları düzelmeli
3. **P1-4** chunk metnini lazy yükleme — istek başına RAM
4. **P0-5** `scripts/reindex.py` + chunking filtreleri — **DB yedeği alarak**

`llm.py` (Foundry Local workaround'ları) bu fazda **hiç açılmayacak**.

Her adımdan sonra: `pytest` + `benchmarks.compare` + `evals.compare`.
Herhangi bir eval vakası bozulursa adım geri alınacak.
