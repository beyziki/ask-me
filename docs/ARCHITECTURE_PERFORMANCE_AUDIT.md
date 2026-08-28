# Ask Me? — Architecture & Performance Audit

**Tarih:** 2026-08-17 · **Kapsam:** `backend/`, `frontend-web/`, `tests/`, `data/` (canlı DB + FAISS)
**Kod değiştirilmedi.** Bu rapor yalnızca analiz + ölçüm sonuçlarıdır.

> **GÜNCELLEME — Faz 1 sonrası (2026-08-17).** Ölçüm altyapısı kurulup testler
> gerçekten çalıştırılınca üç değişiklik oldu. İkisi statik analizde
> görülemeyecek **yeni P0**, biri de bu raporda yazdığım bir iddianın
> **çürütülmesi**:
>
> - **YENİ P0-6** — `.pptx` binary olarak indekslenmiş: owner 1 korpusunun
>   **%29.3'ü çöp**. (bkz. § 1'in sonundaki ek)
> - **YENİ P0-7** — `pytest` üretim veritabanına yazıyor: 9 kullanıcıdan 6'sı
>   test fixture'ı. (bkz. § 1'in sonundaki ek)
> - **P1-5 DÜZELTİLDİ** — "sınırsız önbellek sızıntısı" iddiası **yanlıştı**;
>   ölçüm çürüttü. Aşağıda düzeltildi.

---

## 0. Yöntem — neyi nasıl ölçtüm

Tahmin yürütmemek için senin gerçek verinle ölçtüm: `data/ask_me.db` (43 MB) ve
`data/vectorstore/user_1.faiss` bir sandbox'a kopyalanıp `rag.py`'deki tokenizasyon,
BM25 ve FAISS kod yolları birebir yeniden çalıştırıldı (faiss-cpu 1.15, rank-bm25).

**Ölçüm ortamı uyarısı:** bu sayılar sandbox CPU'sunda alındı, senin makinende
mutlak değerler farklı olacak. Ama *oranlar* ve *ölçekle büyüme eğrisi* aynı —
raporda kararlar bunlara dayanıyor.

### Mevcut durum — baseline ölçümler

| Ölçüm | Değer | Kaynak |
|---|---|---|
| `SELECT` tüm chunk'lar (owner=1, 1014 satır / 4.46 MB metin) | **9.7 ms** | `ask.py:121` |
| `SELECT` doküman filtreli (698 satır, index yok) | 3.0 ms | `ask.py:120` |
| **BM25 korpus tokenizasyonu (her soruda tekrar)** | **379.3 ms** | `rag.py:258` |
| BM25Okapi index kurulumu (cache'li, ilk soru) | 200.1 ms | `rag.py:246` |
| BM25 `get_scores` (sorgu başına) | 0.7 ms | `rag.py:263` |
| FAISS `read_index` diskten | 0.3 ms | `rag.py:81` |
| FAISS `search` top_k=8 (1014 vektör, IndexFlatIP) | **0.06 ms** | `rag.py:206` |

### Ölçekle büyüme (aynı korpus çoğaltılarak)

| Korpus | Tokenizasyon | BM25 kurulum | get_scores |
|---|---|---|---|
| 1.014 chunk (bugün) | 379 ms | 200 ms | 0.7 ms |
| 10.140 chunk (10×) | **6.727 ms** | 2.560 ms | 10 ms |
| 50.700 chunk (50×) | **31.031 ms** | 12.807 ms | 56 ms |

**Okunuşu:** FAISS ve BM25 skorlaması bu ölçekte tamamen ihmal edilebilir
(0.06 ms + 0.7 ms). LLM dışındaki tüm gecikme **tek bir satırda** toplanıyor:
her soruda tekrarlanan korpus tokenizasyonu. 10× büyümede bu tek satır soru
başına 6.7 saniye ekliyor.

### Veri kalitesi ölçümleri (owner_id=1, 1014 chunk)

| Ölçüm | Değer |
|---|---|
| Ortalama chunk uzunluğu | 2.242 karakter |
| En uzun chunk | **13.200 karakter** |
| >3000 karakterlik chunk sayısı | **448 / 1014 (%44)** |
| <50 karakterlik chunk | 101 |
| Tamamen boş chunk | **6** |
| Birebir aynı içerikli chunk (owner=1) | 0 |
| Birebir aynı içerikli chunk (tüm kullanıcılar) | 18 |
| SQLite `page_count` / `freelist_count` | 10.538 / **9.081 (%86 boş)** |
| `journal_mode` | `delete` (WAL **değil**) |

---

## 1. Bulgular

Her bulgu: **Problem → Neden problem → Nerede → Etki → Tahmini kazanç → Risk → Öncelik → Çözüm**

---

### P0-1 · `document_ids` filtresi semantic search'ü fiilen kapatıyor

**Problem.** Kullanıcı "belirli dosyalarda ara" seçtiğinde semantic (FAISS)
tarafı neredeyse hiç sonuç döndürmüyor.

**Neden problem.** FAISS index kullanıcının **tüm** chunk'larını içeriyor.
`semantic_search` `top_k=8` ile **tüm index'te** arıyor, dönen satırları sonra
`row_to_chunk_id` sözlüğüyle (yalnızca seçili dokümanların satırları) filtreliyor.
Seçilmeyen dokümanlara düşen hit'ler sessizce **atılıyor** — yerine yenisi
alınmıyor.

**Nerede.** `backend/app/api/ask.py:118-136` + `backend/app/services/rag.py:197-214`

```python
query = db.query(Chunk).filter(Chunk.owner_id == user_id)
if payload.document_ids:
    query = query.filter(Chunk.document_id.in_(payload.document_ids))
chunks = query.all()
row_to_chunk_id = {c.vector_row: c.id for c in chunks if c.vector_row is not None}
...
sem_hits = semantic_search(user_id, payload.question, settings.top_k_semantic, row_to_chunk_id)
```

**Etki (senin gerçek verinle ölçüldü, user 1):**

| Seçilen doküman | Chunk sayısı | Korpus payı | top_k=8'den beklenen hayatta kalan hit |
|---|---|---|---|
| doc 7 | 698 | %69 | ~5.5 |
| doc 2 | 297 | %29 | ~2.3 |
| doc 30 | 17 | %2 | **~0.1** |
| doc 16 | 2 | %0.2 | **~0.0** |

Zincirleme sonuç üç kat derine iniyor:
1. Semantic sonuç ~yok → cevap fiilen **yalnızca BM25**'e dayanıyor (kelime eşleşmesi;
   eş anlamlı/kavramsal sorularda çuvallıyor).
2. `best_semantic ≈ 0` → `min_relevance_score = 0.30` eşiği **geçilemiyor**
   (`ask.py:145`) → `has_context = False`.
3. Frontend "Dosyalarında bu soruyla ilgili bir bölüm bulunamadı" uyarısı
   gösteriyor (`AskPage.tsx:276`) — oysa bölüm var, arama onu bulamadı.

Bu, 13 Ağustos'ta bildirdiğin "asistan kötü cevap veriyor" şikâyetinin en güçlü
tek teknik açıklaması.

**Tahmini kazanç.** Doküman-filtreli sorularda retrieval recall'ü ~0'dan tam
seviyeye. Latency etkisi nötr (FAISS zaten 0.06 ms).

**Risk.** Düşük–orta. FAISS `IDSelector` API'si `IndexFlatIP`'te destekleniyor
ama sürüme duyarlı; fallback olarak over-fetch yolu her sürümde çalışır.

**Öncelik.** **P0** — doğruluk hedefinin (#2) tek en büyük kalemi.

**Çözüm.**
- Birincil: `faiss.SearchParametersIVF`/`SearchParameters(sel=faiss.IDSelectorBatch(rows))`
  ile aramayı seçili satırlara kısıtla.
- Fallback (sürümden bağımsız, 3 satır): `effective_k = min(index.ntotal, ceil(top_k * index.ntotal / len(row_to_chunk_id)) + top_k)`
  ile over-fetch yapıp filtreden sonra `top_k`'ya kırp.
- Uzun vade: doküman-bazlı `IndexIDMap2` ile gerçek metadata filtreleme.

---

### P0-2 · BM25 korpusu her soruda yeniden tokenize ediliyor (ölçüldü: 379 ms)

**Problem.** `_get_bm25` cache'i var ama **by-pass ediliyor**: cache'e
bakılmadan önce tüm korpus zaten tokenize edilmiş oluyor.

**Neden problem.** Filtre adımı (`query_token_set.intersection(tokens)`) her
chunk'ın token listesine ihtiyaç duyuyor ve bunu cache'ten değil sıfırdan
hesaplıyor. Yani `_get_bm25`'in kazandırdığı 200 ms'nin karşılığında 379 ms
kaybediliyor — net kayıp.

**Nerede.** `backend/app/services/rag.py:254-262`

```python
chunk_ids = list(chunk_id_to_text.keys())
corpus_tokens = [tokenize(chunk_id_to_text[cid]) for cid in chunk_ids]   # <-- 379 ms, HER SORUDA
bm25 = _get_bm25(chunk_ids, [chunk_id_to_text[cid] for cid in chunk_ids]) # <-- cache'li
```

**Etki.** Soru başına sabit **379 ms** (bugün). 10× korpusta **6.7 s**, 50×'te **31 s**.
Bu süre `_retrieve` içinde, yani ilk SSE olayı (`sources`) gönderilmeden önce
harcanıyor → doğrudan **time-to-first-token**'a ekleniyor. Ayrıca tamamen
CPU-bound ve GIL tutuyor → eşzamanlı diğer istekleri de yavaşlatıyor.

**Tahmini kazanç.** TTFT'de **−379 ms** (bugün), 10× ölçekte **−6.7 s**.
Bu, LLM dışındaki en büyük tek kazanç.

**Risk.** Çok düşük. Cache anahtarı (`chunk_ids` imzası) zaten mevcut ve doğru.

**Öncelik.** **P0**

**Çözüm.** `_get_bm25` cache'ine token setlerini de koy:
`_bm25_cache[signature] = (chunk_ids, bm25, [set(t) for t in corpus_tokens])`.
Bellek maliyeti ~korpusun 1/3'ü (bugün ~1.5 MB, kabul edilebilir).

---

### P0-3 · Dosya yükleme tüm backend'i kilitliyor

**Problem.** `upload_document` `async def` olarak tanımlı ama içinde tamamen
**bloklayan** CPU işi yapıyor.

**Neden problem.** FastAPI'de `async def` path operation'ları **event loop'ta**
çalışır. `process_upload` (pypdf parse) ve `add_chunks_to_index`
(sentence-transformers encode + FAISS write) senkron. Bu süre boyunca event loop
tamamen duruyor: `/health` bile cevap vermiyor, diğer kullanıcıların açık
`/ask/stream` bağlantılarına token akmıyor.

**Nerede.** `backend/app/api/documents.py:27-69`

```python
async def upload_document(...):
    dest.write_bytes(await file.read())
    file_type, chunks = process_upload(dest, file.filename)   # pypdf, bloklar
    ...
    vector_rows = add_chunks_to_index(user_id, chunks)        # embedding, bloklar
```

Projedeki **diğer tüm** endpoint'ler `def` (threadpool) — sorun yalnızca burada.

**Etki.** `data/uploads/2/hafta 5.pdf` 7.7 MB. Böyle bir dosyanın parse +
embedding'i CPU'da onlarca saniye. O sürede uygulama tamamen donuk.
Bu, hedef #4 ("ingestion sırasında UI bloklanması") ve hedef #6 (çoklu kullanıcı)
ihlalinin doğrudan sebebi.

**Tahmini kazanç.** Yükleme sırasında diğer isteklerin latency'si "donuk"tan
normale. Yüklemenin kendi süresi değişmez.

**Risk.** Çok düşük — `async` → `def` tek kelimelik değişiklik, `await file.read()`
yerine `file.file.read()`.

**Öncelik.** **P0**

**Çözüm.** İki aşamalı:
1. Hemen: `async def` → `def` (threadpool'a taşı). Tek satır, davranış aynı.
2. Sonra: gerçek arka plan job — `Document.status` sütunu (`pending`/`indexing`/`ready`),
   upload anında 202 dön, ingestion'ı worker thread'de yap, frontend'de
   ilerleme göster. Büyük dosyalarda UI'ın hiç beklememesi için gerekli.

---

### P0-4 · Kimlik doğrulama yok

**Problem.** Kimlik yalnızca istemcinin gönderdiği `X-User-Id` header'ından
okunuyor; doğrulama yok.

**Neden problem.** `curl -H "X-User-Id: 1"` ile herkes 1 numaralı kullanıcının
**tüm dosyalarını, özetlerini, quizlerini** okuyabilir ve **silebilir**.
`/users/login` bir token dönmüyor, sadece kullanıcı nesnesi. Yani "çok kullanıcılı"
mimari fiilen mevcut değil — yalnızca UI seviyesinde ayrım var.

**Nerede.** `backend/app/api/deps.py:9-16`, `backend/app/api/users.py:30-35`,
`frontend-web/src/api/client.ts:setUserId`

```python
def get_current_user_id(x_user_id: int = Header(...)) -> int:
    if x_user_id <= 0:
        raise HTTPException(status_code=401, ...)
    return x_user_id
```

**Etki.** Tam yatay yetki aşımı (IDOR). Kodda `TODO (Hafta 3): JWT` notu var,
`python-jose[cryptography]` requirements'ta duruyor ama hiç import edilmiyor.

**Tahmini kazanç.** Performans kazancı yok — **production-quality hedefinin (#8)
ön koşulu.**

**Risk.** Orta — frontend'in login/refresh akışı ve `postSSE`'nin header
mantığı birlikte değişmeli.

**Öncelik.** **P0** (güvenlik).

**Çözüm.** `/users/login` → kısa ömürlü JWT + refresh; `get_current_user_id`
token'ı doğrulasın; axios interceptor + `postSSE`'de `Authorization` header'ı.
`python-jose` zaten kurulu.

---

### P0-5 · Chunk korpusu karışık: %44'ü eski, dev boyutlu

**Problem.** `chunk_size` 500 kelimeden 220'ye düşürüldü ama **mevcut dosyalar
yeniden işlenmedi**. Korpus iki farklı rejimin karışımı.

**Neden problem.** Ölçüm (owner=1): ortalama 2.242 karakter, **max 13.200**,
**448/1014 chunk 3000 karakterin üzerinde**. Beklenen (220 kelime) ≈ 1.400 karakter.

Üç ayrı zarar:
1. **Context bütçesi yenilyor.** `max_context_chars = 6000`. 3000+ karakterlik
   iki chunk bütçeyi bitiriyor → `_cap_context` (`ask.py:83`) `max_context_chunks=5`
   yerine fiilen **2 chunk** bırakıyor. Recall düşüyor.
2. **Embedding bulanıklaşıyor.** 13.200 karakterlik bir chunk'ın tek vektörü
   birçok konunun ortalaması. Kosinüs benzerliği düşük çıkıyor →
   `min_relevance_score=0.30` eşiği **yanlışlıkla** tetikleniyor → gereksiz
   "kaynaklarda bulunamadı" uyarısı.
3. **Prefill süresi.** Uzun chunk = uzun prompt = uzun TTFT.

Ayrıca **101 chunk 50 karakterden kısa, 6 tanesi tamamen boş.** Bunlar
embedding'lenip FAISS'e yazılmış; boş metin vektörü rastgele bir yöne bakıyor ve
alakasız sorularda üst sıraya çıkabiliyor. `ingestion.py:chunk_text` boş/çöp
parça elemesi yapmıyor.

**Nerede.** `backend/app/services/ingestion.py:41-64`, `backend/app/core/config.py:77-85`

**Etki.** Doğrudan hedef #2 (RAG doğruluğu) ve hedef #1 (hız).

**Tahmini kazanç.** Yeniden indeksleme sonrası context'e giren parça sayısı
2→5'e çıkar (recall), her parça daha odaklı olur (precision).
Ölçülebilir hedef: retrieval precision@5'te belirgin artış (Faz 1'de
oluşturulacak eval seti ile doğrulanacak).

**Risk.** Orta — yeniden indeksleme tüm dosyaları yeniden embed'ler (1014 chunk,
CPU'da dakikalar). Geri alınabilir değil (eski chunk'lar gider) → önce DB yedeği.

**Öncelik.** **P0**

**Çözüm.**
- `scripts/reindex.py`: mevcut dokümanları diskteki orijinal dosyadan yeniden
  parse et, güncel `chunk_size` ile chunk'la, FAISS'i baştan kur.
- `chunk_text`'e minimum uzunluk filtresi (ör. <30 karakter parçaları at) +
  paragraf/cümle sınırına saygılı bölme.
- İdeali: kelime bazlı sliding-window yerine **yapı-farkında** chunking
  (PDF'te başlık/paragraf, kodda fonksiyon). `ingestion.py:46`'daki not zaten
  bunu öngörmüş.

---

### P0-6 · `.pptx` binary olarak indekslenmiş — korpusun %29'u çöp *(Faz 1'de bulundu)*

**Problem.** Desteklenmeyen bir dosya tipi sessizce "metin" sayılıp ham
baytları indekslenmiş.

**Neden problem.** `detect_file_type` yalnızca `.pdf`, `.md` ve kod
uzantılarını tanıyor; **diğer her şey** `"text"` dönüyor ve
`path.read_text(encoding="utf-8", errors="ignore")` ile okunuyor. `.pptx`,
`.docx`, `.xlsx` hepsi ZIP arşivi — bu, sıkıştırılmış baytları "metin" diye
kabul etmek demek.

```
.pptx -> text    .docx -> text    .xlsx -> text    .zip -> text    .png -> text
```

**Nerede.** `backend/app/services/ingestion.py:22-38`

**Etki (ölçüldü).** `Chapter_6.pptx` → **297 chunk mojibake**, owner 1
korpusunun **%29.3'ü**. Chunk #0'ın gerçek içeriği: `PK` (ZIP imzası).
Bu 297 chunk hem FAISS index'inde hem BM25 korpusunda duruyor; her soruda
rastgele token eşleşmeleriyle yarışıyor ve `min_relevance_score` eşiğini
bozuyor. Kullanıcıya **hiçbir hata gösterilmiyor** — dosya "başarıyla
yüklendi" görünüyor.

Çöplük oranı ölçümü (yazdırılamayan karakter oranı, doküman başına):

| doc | dosya | chunk | ort. çöplük |
|---|---|---|---|
| 2 | Chapter_6.pptx | 297 | **%26.7** |
| 7 | Computer Networking (PDF) | 698 | %0.1 |
| 30 | BabyRobot (PDF) | 17 | %0.1 |

**Tahmini kazanç.** Owner 1 için korpusun %29'u gürültüden temizlenir —
doğrudan hedef #2.

**Risk.** Düşük. **Öncelik.** **P0** — Faz 4'e eklendi.

**Çözüm.** (1) `detect_file_type` bilinmeyen uzantıyı reddetsin (400), sessizce
`"text"`e düşmesin. (2) `.pptx`/`.docx` gerçekten desteklensin
(`python-pptx`/`python-docx`) — kullanıcı zaten pptx yüklemiş, ihtiyaç var.
(3) Ingestion'a "çıkarılan metin okunabilir mi?" kontrolü (yazdırılamayan
karakter oranı eşiği), ki başka bir format sessizce sızmasın.

---

### P0-7 · `pytest` üretim veritabanına yazıyor *(Faz 1'de bulundu, düzeltildi)*

**Problem.** Testler `data/ask_me.db`'ye kullanıcı ve doküman yazıyor, hatta
**siliyordu**.

**Neden problem.** `settings` **modül import zamanında** bir kez oluşuyor.
Bazı test dosyaları import etmeden önce `DATABASE_URL` ayarlıyordu
(`test_multiuser`, `test_document_groups`, `test_rag_index`,
`test_summary_utils`, `test_ask_retrieval`), bazıları **ayarlamıyordu**
(`test_ingestion`, `test_llm_utils`, `test_quiz_utils`, `test_rag_merge`).
pytest modülleri alfabetik topladığı için, ayarlamayan bir modül önce
yüklendiğinde `settings` gerçek veritabanına bağlanıyor ve sonraki tüm
modüllerin env yazması **etkisiz** kalıyordu.

**Kanıt.** `import tests.test_ingestion` →
`settings.database_url = sqlite:////.../data/ask_me.db`.
Ve üretim veritabanındaki 9 kullanıcıdan **6'sı test fixture'ı**:
`alice_docs`, `alice_ml`, `bob_docs`, `carol_en`, `carol_tr`, `dup_user`.
Gerçek kullanıcılar yalnızca `byzerdem`, `beyza`, `ben`.

**Etki.** Sadece kirlilik değil:
`test_delete_document_removes_it_and_ungroups_related_quiz` **doküman
siliyordu**. DB'nin %86'sının boş alan olması (P1-3) muhtemelen kısmen bundan.

**Öncelik.** **P0** — **Faz 1'de düzeltildi** (`tests/conftest.py`).

**Çözüm (uygulandı).** `conftest.py` env değişkenlerini her test modülünden
önce ayarlıyor; ayrıca oturum başında izolasyonu **doğrulayan** bir fixture
var — düzen ileride bozulursa testler sessizce üretime yazmak yerine anında
kırmızıya döner.

---

### P1-1 · Foundry Local'e eşzamanlı istek koruması yok

**Problem.** İki kullanıcı aynı anda soru sorarsa iki istek aynı anda Foundry
Local'e gidiyor.

**Neden problem.** 8 GB VRAM'de KV cache ikiye katlanıyor. `llm.py:102-120`
zaten belgeliyor: bu durumda `CUDA illegal memory access` geliyor ve bu hata
**süreç içinde kurtarılamıyor** — bağlam zehirleniyor, sonraki *her* istek de
patlıyor. Yani tek bir eşzamanlı çift, backend'i Foundry servisi elle yeniden
başlatılana kadar kullanılamaz hale getirebiliyor.

**Nerede.** `backend/app/services/llm.py:874-964`, `services/quiz.py`, `services/summary.py` —
hiçbirinde eşzamanlılık sınırı yok.

**Etki.** Hedef #6'nın (çoklu kullanıcı) en kritik engeli. Sınıf ortamında
(summer school) aynı anda birden fazla öğrenci = kaçınılmaz.

**Tahmini kazanç.** Latency kazancı yok — **çökmeyi önlüyor.** Tek GPU'da
seri çalıştırmak zaten toplam throughput'u düşürmez (GPU zaten tek istekle
doyuyor).

**Risk.** Düşük.

**Öncelik.** **P1**

**Çözüm.** LLM çağrılarını saran `threading.BoundedSemaphore(settings.llm_concurrency)`
(varsayılan 1). Kuyrukta bekleyen istemciye SSE `{"type":"queued","position":N}`
olayı gönder; frontend "sırada 2. kişisin" göstersin. Ayrıca semafor bekleme
süresine timeout.

---

### P1-2 · SQLite WAL modunda değil + eksik index'ler

**Problem.** `journal_mode = delete`. Yazan tek bir istek süresince okuyucular
bloklanıyor.

**Neden problem.** SQLite rollback-journal modunda yazma, **veritabanı dosyasının
tamamını** kilitliyor. `pool_size=10, max_overflow=20` (`db/base.py:15`) bunu
çözmez — kilit havuzda değil, dosyada. `ask.py:132` ve `quiz.py:109`'daki
`db.close()` düzeltmeleri doğru yönde ama kök nedeni ortadan kaldırmıyor.

Eksik index'ler:
- `chunks.document_id` — **yok**. `/summary`, `/quiz` ve doküman-filtreli `/ask`
  bu sütunla filtreliyor (`summary.py:113`, `quiz.py:82`, `ask.py:120`).
- `documents.owner_id` — **yok**. `/documents` listesi her açılışta tam tarama.
- `quizzes.owner_id`, `quizzes.document_id` — **yok**.

**Nerede.** `backend/app/db/base.py:7-19`, `backend/app/db/models.py`

**Etki.** Bugün küçük (16 doküman); ölçekte hızla kötüleşiyor. WAL eksikliği
ise **bugün bile** hissediliyor: bir upload sürerken login/soru bekleme yapıyor.

**Tahmini kazanç.** WAL: eşzamanlı okuma/yazma bekleme süresi ~0'a.
Index'ler: doküman-filtreli sorguda tam tarama → index seek.

**Risk.** Çok düşük. WAL geri alınabilir; index eklemek şemayı bozmaz.

**Öncelik.** **P1**

**Çözüm.**
```python
@event.listens_for(engine, "connect")
def _pragmas(conn, _):
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA busy_timeout=5000")
    cur.close()
```
+ `run_startup_migrations`'a `CREATE INDEX IF NOT EXISTS` satırları.

**PostgreSQL gerekli mi?** **Hayır.** Proje tamamen offline, tek makine, tek
process. WAL + doğru index'ler bu iş yükü için fazlasıyla yeterli. PostgreSQL'e
geçiş yalnızca (a) birden fazla backend process'i veya (b) ağ üzerinden paylaşılan
kurulum gerekirse anlamlı olur. Şimdi geçmek, kazancı olmayan bir kurulum
karmaşıklığı ekler. **Öneri: SQLite'ta kal, WAL'a geç.**

---

### P1-3 · Veritabanının %86'sı boş alan

**Problem.** `page_count = 10.538`, `freelist_count = 9.081`. 43 MB dosyanın
yalnızca ~6 MB'ı gerçek veri.

**Neden problem.** Silinen dokümanların sayfaları geri kazanılmıyor
(`auto_vacuum` kapalı, hiç `VACUUM` çalıştırılmamış).

**Etki.** Yedekleme boyutu, açılış I/O'su, page cache verimliliği. Fonksiyonel
bir hata değil ama 43 MB'lık bir dosyayı "bu proje ne kadar veri tutuyor?"
sorusuna cevap olarak göstermek yanıltıcı.

**Tahmini kazanç.** 43 MB → ~6 MB (%86 küçülme).

**Risk.** `VACUUM` süresince DB kilitli (bu boyutta saniyeler).

**Öncelik.** **P1** (kolay kazanç)

**Çözüm.** Tek seferlik `VACUUM` + `PRAGMA auto_vacuum=INCREMENTAL` +
`delete_document`'ta periyodik `PRAGMA incremental_vacuum`.

---

### P1-4 · `_retrieve` her soruda tüm chunk metnini belleğe çekiyor

**Problem.** `ask.py:121` `chunks = query.all()` → 1014 ORM nesnesi, 4.46 MB metin,
her soruda.

**Neden problem.** Bu metnin tamamına aslında ihtiyaç yok:
- BM25 index'i zaten cache'li (`_get_bm25`),
- token setleri de cache'lenebilir (bkz. P0-2),
- gerçekten gereken tek şey **kazanan 5 chunk'ın** metni.

**Nerede.** `backend/app/api/ask.py:118-125`

**Etki.** Bugün 9.7 ms DB + ORM nesne kurulumu + ~45 MB geçici RAM. 10× korpusta
istek başına ~450 MB geçici RAM → 5 eşzamanlı istekte OOM riski. Hedef #3 ve #4'ün
kalbi.

**Tahmini kazanç.** İstek başına RAM'de ~10× azalma; ölçekte DB+ORM süresinde
benzer oranda.

**Risk.** Orta — `chunk_id_to_obj` (kaynak listesi için) ve `_dedupe_by_content`
metne bağlı; ikisinin de yeniden yazılması gerekiyor.

**Öncelik.** **P1**

**Çözüm.** İki katmanlı:
1. DB'den yalnızca `(id, document_id, chunk_index, vector_row)` çek
   (`with_entities`) — metin yok.
2. Metinleri `chunk_id → text` process-level LRU cache'te tut (BM25 cache'iyle
   aynı imzayı paylaşabilir), yalnızca cache miss'te DB'ye git.
3. `sources` için gereken `filename` zaten ayrı sorguyla alınıyor — orada değişiklik yok.

---

### P1-5 · FAISS index cache'inin üst sınırı yok *(ölçümle düzeltildi)*

> **DÜZELTME (2026-08-17).** Bu maddede önce şunu yazmıştım: *"Anahtar
> `(mtime, size)` içerdiği için her yüklemede yeni bir girdi oluşuyor, eski
> `faiss.Index` bellekte kalıyor — sınırsız sızıntı."* **Bu yanlıştı.**
>
> `benchmarks/bench_resources.py` ile ölçtüm: aynı dosyaya 10 kez yazıp 10 kez
> okuduktan sonra önbellekte **1 girdi** var, 10 değil. Sebep basit —
> `_read_index_cached` önbellek **anahtarı** olarak `str(path)` kullanıyor
> (`rag.py:76`); `(mtime, size)` yalnızca **değerin** bir parçası, geçerlilik
> kontrolü için. Aynı yola tekrar yazmak girdiyi **üzerine yazıyor**.
>
> Sızıntı yok. Aşağısı, ölçümden sonra kalan gerçek (ve daha küçük) sorun.

**Problem.** `_index_cache` (`rag.py:69`) hiçbir boyut sınırına sahip değil.

**Neden problem.** Önbellek **kullanıcı sayısıyla** büyüyor (her kullanıcının
kendi index dosyası, dolayısıyla kendi anahtarı var) ve hiç tahliye edilmiyor.
Ölçüm: 10 farklı kullanıcı index'i → 11 girdi, **+28 MB**. 2000 vektörlük
küçük index'lerle. Sınıf ortamında 30 öğrenci ve gerçek boyutlu index'lerle
bu birkaç yüz MB'a çıkar ve **hiçbir zaman geri verilmez**.

Ayrıca `add_chunks_to_index` (`rag.py:87-110`) cache'i hiç kullanmıyor
(`_load_or_create_index` doğrudan `faiss.read_index`) ve **tüm index'i** her
yüklemede diske yeniden yazıyor — 50k chunk'ta yükleme başına 77 MB yazma.

`_bm25_cache` de LRU değil: 32'ye ulaşınca `.clear()` ile **tamamen** boşalıyor
(`rag.py:249`), yani 33. kullanıcı gelince ilk 32'sinin cache'i uçuyor. Ölçüm:
tek kullanıcının BM25 önbelleği (1014 chunk) **85 MB** — 32 kullanıcı sınırı
bu boyutta gerçekçi değil.

**Nerede.** `backend/app/services/rag.py:63-110, 231-251`

**Etki.** Kullanıcı sayısıyla sınırsız RAM artışı (hedef #3).

**Tahmini kazanç.** Sabit RAM tavanı; büyük korpuslarda upload I/O'sunda azalma.

**Risk.** Düşük.

**Öncelik.** **P1**

**Çözüm.** `_index_cache` ve `_bm25_cache` için gerçek LRU
(`collections.OrderedDict` veya `cachetools.LRUCache`), yazma sonrası
`_index_cache.pop(path)`, `add_chunks_to_index`'te cache'li index'e ekleme
(`IndexFlatIP.add` in-place, dosyaya periyodik flush).

---

### P1-6 · Streaming endpoint'ler threadpool worker'ı tutuyor

**Problem.** `ask_stream`/`create_quiz_stream`/`create_summary_stream` `def`
(senkron) ve `StreamingResponse`'a **senkron generator** veriyor.

**Neden problem.** Starlette senkron generator'ı `anyio` threadpool'unda iterate
ediyor. Yani her aktif SSE bağlantısı, üretim boyunca (dakikalar sürebiliyor)
bir threadpool worker'ı işgal ediyor. Varsayılan havuz 40 worker; aynı anda 40
açık stream'de **tüm** endpoint'ler (upload, login, /documents) tıkanıyor.

**Nerede.** `backend/app/api/ask.py:237-320`, `quiz.py:176`, `summary.py:163`

**Etki.** Hedef #6'nın ikinci engeli. Bugün 9 kullanıcıyla görünmüyor, sınıf
ortamında görünür.

**Tahmini kazanç.** Eşzamanlı stream tavanı 40'tan pratikte sınırsıza.

**Risk.** Orta — generator'ı `async` yapmak, LLM çağrısını
`anyio.to_thread.run_sync` ile bir kuyruğa bağlamayı gerektiriyor.
**Foundry Local workaround'larına dokunulmamalı** (`llm.py` içi aynen kalmalı;
yalnızca çağrı sarmalayıcısı değişir).

**Öncelik.** **P1**

**Çözüm.** `async def` endpoint + `async` generator; senkron LLM generator'ını
`anyio.to_thread` + `memory_object_stream` köprüsüyle tüket. `llm.py` hiç
değişmez.

---

### P1-7 · İstemci bağlantıyı kesince üretim devam ediyor

**Problem.** Kullanıcı sekmeyi kapatır veya "yeni soru" sorarsa eski üretim
GPU'da çalışmaya devam ediyor.

**Neden problem.** Backend'de `request.is_disconnected()` kontrolü yok.
Frontend'de `postSSE` bir `AbortSignal` parametresi **kabul ediyor**
(`endpoints.ts:115`) ama `AskPage`/`QuizPage`/`SummaryPage` hiç göndermiyor →
"Durdur" butonu yok, iptal edilemiyor.

**Nerede.** `backend/app/api/ask.py:254`, `frontend-web/src/pages/AskPage.tsx:445`

**Etki.** Boşa GPU/VRAM (hedef #3); P1-1'in semaforuyla birleşince kuyruğu da
gereksiz tıkıyor.

**Tahmini kazanç.** Terk edilen üretimlerde %100 GPU tasarrufu.

**Risk.** Düşük.

**Öncelik.** **P1**

**Çözüm.** Backend: her token'dan sonra `await request.is_disconnected()` →
generator'ı kapat. Frontend: `AbortController`'ı `useRef`'te tut, yeni soru
gönderirken/unmount'ta `abort()`, UI'a "Durdur" butonu.

---

### P1-8 · `_get_model_id` bayat cache tutuyor, kurtarma yolu yok

**Problem.** `@lru_cache(maxsize=1)` ile model id süreç ömrü boyunca sabitleniyor.

**Neden problem.** Foundry Local servisi yeniden başlatılırsa (ki `GpuContextLost`
mesajı kullanıcıya tam olarak bunu söylüyor, `llm.py:63-69`) model bellekten
düşüyor. Cache'li id ile yapılan her istek "Model is not loaded" alıyor ve
**backend süreci yeniden başlatılana kadar** düzelmiyor. Yani `scripts/foundry-doctor.ps1`
çalıştırmak tek başına yetmiyor — kullanıcı bunu bilmiyor.

**Nerede.** `backend/app/services/llm.py:292-324`

**Etki.** Gözlemlenen "GPU bağlam kaybı" senaryosundan sonra tam kurtarma
imkânsız.

**Tahmini kazanç.** Kurtarma süresi "backend restart"tan otomatiğe.

**Risk.** Düşük.

**Öncelik.** **P1**

**Çözüm.** `_get_model_id.cache_clear()` çağıran bir `reset_model_cache()` +
`GpuContextLost`/"not loaded" hatası yakalandığında otomatik çağrı + bir kez
yeniden deneme. Ayrıca `POST /health/model/reload` ucu.

---

### P2-1 · Reranking yok

**Problem.** RRF sonrası doğrudan ilk 5 alınıyor; ikinci bir alaka değerlendirmesi yok.

**Neden problem.** RRF yalnızca **sıra** birleştiriyor, gerçek alakayı ölçmüyor.
Cross-encoder rerank (sorgu+chunk çiftini birlikte kodlar) tipik olarak
precision@5'te belirgin kazanç verir.

**Nerede.** `backend/app/api/ask.py:152`

**Etki / kazanç.** Ölçülmedi — Faz 1'deki eval seti olmadan kazanç iddia etmek
tahmin olur.

**Risk.** Orta. Ek model (~500 MB RAM) + CPU'da 20 aday için ~200-400 ms.
Hedef #1 (hız) ile **doğrudan çelişiyor**.

**Öncelik.** **P2** — ve **koşullu**.

**Çözüm / karar.** Önce P0-1 (doküman filtresi) ve P0-5 (chunking) düzeltilsin,
sonra eval setiyle ölçülsün. Rerank bozuk bir aday listesini kurtarmaz; aday
listesi düzeldikten sonra hâlâ precision açığı varsa, `.env` bayrağı arkasında
(varsayılan **kapalı**) eklenir. Şu anda eklemek, ölçülmemiş bir kazanç için
kesin bir gecikme satın almak olur.

---

### P2-2 · Sorgu embedding'i cache'lenmiyor

**Problem.** `semantic_search` her çağrıda `embedder.encode([query])`.

**Nerede.** `backend/app/services/rag.py:203`

**Etki.** MiniLM-L12 CPU'da tek cümle için ~15-30 ms. Küçük ama bedava.
`llm.py` yorumlarında "kullanıcı aynı soruyu art arda 3 kez gönderdi" gözlemi
kayıtlı — tekrar eden sorularda tam tasarruf.

**Kazanç.** Tekrar eden sorularda −15-30 ms.
**Risk.** Yok. **Öncelik.** **P2**
**Çözüm.** `@lru_cache(maxsize=512)` ile sorgu → vektör.

---

### P2-3 · `get_embedder()` thread-safe değil

**Problem.** Global `_embedder`, kilit yok.

**Nerede.** `backend/app/services/rag.py:26-30`

**Etki.** İki thread aynı anda ilk çağrıyı yaparsa model iki kez yükleniyor
(~500 MB × 2 RAM). Warmup bunu pratikte önlüyor ama `WARMUP_ON_STARTUP=false`
(testler/CI, `config.py:52`) ile yarış açık.

**Kazanç.** Kenar durumda −500 MB. **Risk.** Yok. **Öncelik.** **P2**
**Çözüm.** `threading.Lock` ile double-checked locking.

---

### P2-4 · `_dedupe_by_content` yalnızca birebir eşitlik yakalıyor

**Problem.** `seen: set[str]` tam metin karşılaştırması yapıyor.

**Nerede.** `backend/app/api/ask.py:56-80`

**Etki.** Ölçüm: owner=1'de birebir aynı chunk **0**, global 18. Yani bugün bu
fonksiyon **pratikte hiç iş yapmıyor**. Asıl problem ise yakalanmıyor:
`chunk_overlap=40` kelime yüzünden komşu chunk'lar ~%18 örtüşüyor ve ikisi de
context'e girebiliyor — LLM'e aynı cümleleri iki kez göndermek hem token hem
dikkat israfı.

**Kazanç.** Context'te ~%10-18 tekrar eliminasyonu → daha kısa prefill.
**Risk.** Orta — agresif dedupe gerçek farklı içeriği de eleyebilir.
**Öncelik.** **P2**
**Çözüm.** Normalize edilmiş (boşluk/küçük harf) hash + shingle-tabanlı Jaccard
eşiği (ör. >0.8 → aynı say). Eşik eval setiyle ayarlanmalı.

---

### P2-5 · Frontend: kod bölme yok

**Problem.** Tüm sayfalar `App.tsx`'te statik import.

**Nerede.** `frontend-web/src/App.tsx:5-11`

**Etki.** `react-markdown` + `remark-gfm` (~120 kB gzip, unified/mdast zinciriyle
birlikte) **LoginPage'de bile** indiriliyor. İlk açılış (hedef #5) doğrudan
etkileniyor.

**Kazanç.** İlk bundle'da tahminen %40-50 azalma — Faz 1'de `vite build`
çıktısıyla ölçülecek.
**Risk.** Düşük. **Öncelik.** **P2**
**Çözüm.** Route bazlı `React.lazy` + `Suspense`; `Markdown` bileşenini de lazy yap.

---

### P2-6 · Frontend: her token'da tüm ağaç yeniden render ediliyor

**Problem.** `onToken` → `setStreamingText(liveText)` → `AskPage` tamamen render;
`Markdown` **tüm** metni her token'da yeniden parse ediyor.

**Nerede.** `frontend-web/src/pages/AskPage.tsx:458-461`, `components/Markdown.tsx`

**Etki.** 800 token'lık bir cevapta 800 kez tam markdown parse + tüm mesaj
listesi + `DocumentPicker` render. Uzun cevaplarda gözle görülür takılma
(hedef #7).

Yan bulgular aynı dosyada:
- `DocumentsContext` provider value her render'da yeni obje →
  tüm tüketiciler gereksiz render (`DocumentsContext.tsx:78`).
- `selectedIds.includes(doc.id)` her dosya satırında O(n) (`AskPage.tsx:185`) →
  `Set` olmalı.
- `messages.map` içindeki mesajlar `React.memo`'lu değil.

**Kazanç.** Streaming sırasında render maliyetinde ~10× azalma (token başına
render yerine ~50 ms throttle).
**Risk.** Düşük — yalnızca render davranışı, iş mantığı değişmiyor.
**Öncelik.** **P2**
**Çözüm.** Streaming metnini ayrı `memo`'lu bileşene taşı; `requestAnimationFrame`
veya 50 ms throttle ile güncelle; `SourcesList`/mesaj balonlarını `memo`'la;
context value'yu `useMemo`'la; `selectedIds`'i `Set` olarak tut.

---

### P2-7 · Frontend: yükleme seri, optimistic UI yok

**Problem.** `UploadPage.handleFiles` dosyaları `for ... await` ile **sırayla**
yüklüyor; liste yalnızca **hepsi bitince** güncelleniyor.

**Nerede.** `frontend-web/src/pages/UploadPage.tsx:78-83`

**Etki.** 5 dosya seçen kullanıcı, 5 dosyanın toplam süresi kadar boş ekrana
bakıyor. Her mutasyondan sonra `refreshDocuments()` **iki** tam liste isteği
atıyor (`/documents` + `/documents/groups`).

**Kazanç.** Algılanan yükleme süresinde belirgin azalma.
**Risk.** Düşük (paralel yükleme backend'i zorlarsa `Promise.all` yerine
sınırlı eşzamanlılık).
**Öncelik.** **P2**
**Çözüm.** Dosya başına `refresh` (veya dönen `DocumentOut`'u optimistic olarak
listeye ekle); grupları yalnızca grup değiştiğinde yeniden çek.

---

### P2-8 · Test kapsamı: kritik yollarda sıfır test

**Problem.** 124 test var — hepsi saf yardımcı fonksiyon seviyesinde.

**Test EDİLEN (iyi durumda):** `strip_think`, `_ThinkStreamStripper`,
`_RepetitionGuard`, `_stream_with_warmup`, `_create_chat_completion` fallback'i,
`build_prompt`, `_parse_quiz_json`, `group_chunks`, `_split_for_retry`,
`tokenize`, `rrf_merge`, `hybrid_merge`, `bm25_search`, `drop_rows_from_index`,
doküman grupları, çok kullanıcılı izolasyon.

**Test EDİLMEYEN:**

| Alan | Durum | Not |
|---|---|---|
| `semantic_search` | **hiç test yok** | P0-1'deki bug tam olarak burada; test olsaydı yakalanırdı |
| `_retrieve` uçtan uca | hiç test yok | RAG'ın kalbi |
| `/ask/stream` SSE | hiç test yok | Ana kullanıcı akışı |
| `/quiz/stream`, `/summary/{id}/stream` | hiç test yok | |
| `add_chunks_to_index`, `rebuild_index` | hiç test yok | yalnızca `drop_rows` test edilmiş |
| Eşzamanlılık / bağlantı havuzu | hiç test yok | P1-1, P1-2, P1-6 |
| **RAG doğruluk (precision/recall)** | **hiç yok** | hedef #2'yi ölçmenin tek yolu |
| **Latency regresyon** | **hiç yok** | hedef #1'i ölçmenin tek yolu |
| Yük testi | hiç yok | hedef #6 |

Ayrıca `conftest.py` yok, `pytest.ini`/`pyproject.toml` yok (testpaths, marker
tanımı yok — her test dosyası kendi `sys.path` ayarını yapıyor).

**Öncelik.** **P2** (ama Faz 1'in ön koşulu — ölçmeden optimize edilemez)

---

### P3 · Küçük ama gerçek hatalar ve borç

| # | Bulgu | Yer | Not |
|---|---|---|---|
| P3-1 | **Aynı dosya iki kez yüklenirse diskte üzerine yazılıyor ama DB'de iki kayıt açılıyor.** Birini silmek diğerinin dosyasını siliyor → sarkan kayıt. | `documents.py:40-52` | Gerçek veride doğrulandı: user 2'de `BabyRobot_IEEE_Report.pdf` ×2, `Turing Machines.pdf` ×2 |
| P3-2 | `file.filename` doğrudan path'e ekleniyor → path traversal (`../../`) | `documents.py:40` | Güvenlik |
| P3-3 | `dest.write_bytes(await file.read())` — 7.7 MB PDF tamamen belleğe | `documents.py:41` | Chunked yazma gerekli |
| P3-4 | `CORS allow_origins=["*"]` | `main.py:24` | Kodda not var, production'da daraltılmalı |
| P3-5 | `_coerce_question_list` `"questions" not in locals()` ile akış kontrolü yapıyor | `quiz.py:130-132` | Kırılgan; `if isinstance(data, list)` sonrası `elif` olmalı |
| P3-6 | `documents.language` sütunu hiç doldurulmuyor (hep NULL) | `models.py:53` | Ölü sütun veya eksik özellik |
| P3-7 | `quiz_questions.source_chunk_id` hiç doldurulmuyor | `quiz.py:_persist_quiz` | "Bu soru hangi parçadan?" özelliği yarım kalmış |
| P3-8 | `db.query(Model).get(id)` SQLAlchemy 2.0'da deprecated | `documents.py:20,104,187`, `quiz.py:94`, `summary.py:120`, `code.py:28` | `db.get(Model, id)` |
| P3-9 | `dt.datetime.utcnow` deprecated (Python 3.12+) | `models.py` (7 yer) | `dt.datetime.now(dt.UTC)` |
| P3-10 | `@app.on_event("startup")` deprecated | `main.py:66` | `lifespan` context manager |
| P3-11 | `Base.metadata.create_all` import-time'da çalışıyor | `main.py:15` | Test izolasyonunu zorlaştırıyor |
| P3-12 | Ölü kod / kullanılmayan bağımlılıklar: `hybrid_merge` + `settings.hybrid_alpha` (yalnızca kendi testleri çağırıyor), `frontend/streamlit_app.py`, requirements'ta hiç import edilmeyen `python-jose`, `tiktoken`, `markdown-it-py`, `streamlit` | `rag.py:321`, `requirements.txt` | `generate_answer`/`generate_quiz`/`generate_summary` (non-stream) **silinmemeli** — testler için değerli |
| P3-13 | `vite.config.ts`'te `usePolling: true` | `vite.config.ts:16` | Docker için gerekli, yerel `npm run dev`'de boşuna CPU yakıyor — koşullu yapılmalı |

---

## 2. Öncelik özeti

| Öncelik | Bulgu | Ana hedef | Tahmini kazanç |
|---|---|---|---|
| **P0-1** | `document_ids` filtresi semantic'i kapatıyor | #2 doğruluk | recall ~0 → tam |
| **P0-2** | BM25 korpus tokenizasyonu tekrarı | #1 hız, #4 ölçek | **−379 ms/soru** (10×'te −6.7 s) |
| **P0-3** | Upload event loop'u kilitliyor | #6 eşzamanlılık | donma → yok |
| **P0-4** | Auth yok (`X-User-Id`) | #8 production | güvenlik |
| **P0-5** | Chunk korpusu karışık (%44 dev, 6 boş) | #2 doğruluk | context 2→5 parça |
| **P0-6** | `.pptx` binary indekslenmiş | #2 doğruluk | owner 1'de korpusun %29'u temizlenir |
| **P0-7** | `pytest` üretim DB'sine yazıyor | #9 test | ✅ Faz 1'de düzeltildi |
| P1-1 | LLM eşzamanlılık koruması yok | #6 | çökme önleme |
| P1-2 | WAL yok + eksik index | #6, #4 | kilit beklemesi ~0 |
| P1-3 | DB %86 boş alan | #3 | 43 MB → ~6 MB |
| P1-4 | Tüm chunk metni belleğe | #3, #4 | ~10× RAM azalması |
| P1-5 | Index/BM25 cache'inin üst sınırı yok *(düzeltildi: sızıntı değil)* | #3 | sabit RAM tavanı |
| P1-6 | SSE threadpool worker'ı tutuyor | #6 | 40 stream tavanı kalkar |
| P1-7 | Disconnect algılanmıyor | #3 | terk edilen üretimde %100 GPU tasarrufu |
| P1-8 | Bayat `model_id` cache'i | #8 | otomatik kurtarma |
| P2-1 | Rerank yok | #2 | **ölçülmeden karar verilmeyecek** |
| P2-2 | Sorgu embedding cache'i yok | #1 | −15-30 ms (tekrarda) |
| P2-3 | `get_embedder` thread-safe değil | #3 | −500 MB (kenar durum) |
| P2-4 | Dedupe near-duplicate yakalamıyor | #2, #1 | context'te %10-18 tekrar |
| P2-5 | Kod bölme yok | #5, #7 | bundle ~%40-50 |
| P2-6 | Token başına tam render | #7 | render ~10× |
| P2-7 | Seri yükleme, optimistic UI yok | #7 | algılanan süre |
| P2-8 | Kritik yollarda test yok | #9 | regresyon güvenliği |
| P3 | 13 küçük bulgu | #8, #10 | — |

---

## 3. Faz planı

Her fazın kuralı: **önce benchmark → değişiklik → test → tekrar benchmark →
önce/sonra karşılaştırması.** Her fazdan önce `pytest` tamamen yeşil olmalı.

### PHASE 1 — Ölçüm altyapısı (kod davranışı değişmez)

Bu faz **hiçbir üretim kodunu değiştirmez**; yalnızca ölçüm ekler.

1. `tests/conftest.py` + `pyproject.toml` (pytest config, `benchmark`/`slow` markerları)
2. `benchmarks/` paketi — her biri JSON çıktı veren, tekrarlanabilir script'ler:
   - `bench_ingestion.py` — PDF parse, chunking, embedding, FAISS write (dosya boyutuna göre)
   - `bench_retrieval.py` — BM25 tokenize/kurulum/skor, FAISS search, `_retrieve` toplam
   - `bench_llm.py` — model warmup, TTFT, toplam üretim, token/s
   - `bench_concurrent.py` — 1/2/5/10 eşzamanlı `/ask/stream`, p50/p95/hata oranı
   - `bench_resources.py` — RSS, VRAM (`nvidia-smi`), CPU (`psutil`)
3. `evals/` — RAG doğruluk seti: senin gerçek PDF'lerinden 30-50 soru +
   beklenen kaynak chunk'lar. Metrikler: **recall@5, precision@5, MRR,
   `has_context` doğruluğu**. Bu, hedef #2'yi ölçmenin tek yolu.
4. `benchmarks/baseline.json` — mevcut kodun tüm sayıları, versiyonlanmış.
5. `npm run build` bundle boyutu raporu (`rollup-plugin-visualizer`).

**Çıktı:** "önce" tablosu. Bundan sonraki her faz bu tabloya karşı ölçülür.

### PHASE 2 — Kritik backend/RAG (P0-1, P0-2, P0-5, P1-4)

Sırayla, her biri ayrı commit + ayrı benchmark:
1. **P0-2** BM25 token cache'i (en yüksek kazanç/risk oranı, tek dosya)
2. **P0-1** doküman filtreli semantic search
3. **P1-4** chunk metni lazy yükleme
4. **P0-5** `scripts/reindex.py` + chunking filtreleri → **eval seti ile önce/sonra**

Foundry Local kodu (`llm.py`) bu fazda **hiç açılmaz**.

### PHASE 3 — LLM latency (P1-1, P1-6, P1-7, P1-8)

Semafor + kuyruk, async SSE köprüsü, disconnect algılama, model cache kurtarma.
`llm.py`'nin **içi** değişmez — yalnızca çağıran katman.
Ölçüm: TTFT, eşzamanlı p95, terk edilen istekte GPU süresi.

### PHASE 4 — Ingestion (P0-3, P0-6, P3-1, P3-2, P3-3)

`async def` → `def`; **desteklenmeyen dosya tipini reddet + `.pptx`/`.docx`
desteği + çıkarılan metnin okunabilirlik kontrolü (P0-6)**; sonra gerçek arka
plan job + `Document.status` + frontend ilerleme; duplicate dosya adı
çakışması; path sanitizasyonu; chunked yazma.
Ölçüm: upload latency, upload sırasında `/health` p95, eval'de owner 1'in
recall'ü (çöp chunk'lar temizlenince).

### PHASE 5 — Frontend (P2-5, P2-6, P2-7, P3-13)

Kod bölme, streaming render throttle, memoization, optimistic upload, iptal
butonu. Ölçüm: bundle boyutu, React Profiler commit sayısı, Lighthouse.

### PHASE 6 — Database (P1-2, P1-3)

WAL + pragma'lar, eksik index'ler, `VACUUM` + `auto_vacuum`.
Ölçüm: eşzamanlı okuma/yazma p95, DB dosya boyutu, sorgu planları (`EXPLAIN QUERY PLAN`).

### PHASE 7 — Test ve regresyon (P2-8)

`semantic_search`, `_retrieve`, SSE endpoint'leri, eşzamanlılık testleri;
CI'da eval + latency eşikleri (regresyonda kırmızı).

### PHASE 8 — Production hardening (P0-4, P3-*)

JWT, CORS daraltma, structured logging + request id, `/metrics`, deprecated
API temizliği, ölü kod ve kullanılmayan bağımlılık temizliği, README güncellemesi.

---

## 4. Açık kararlar

Kod yazmaya başlamadan önce netleşmesi gerekenler:

1. **PostgreSQL?** Önerim **hayır** — SQLite + WAL bu iş yükü için yeterli.
   Katılmıyorsan söyle, Faz 6'yı ona göre planlarım.
2. **Rerank?** Önerim: Faz 2 bitene ve eval seti ölçüm verene kadar **erteleyelim**.
   Hız hedefinle çelişiyor; ölçmeden eklemek tahmin olur.
3. **Yeniden indeksleme (P0-5)** mevcut chunk'ları siler. Önce `ask_me.db` ve
   `data/vectorstore/` yedeğini alacağım — onay ister misin, yoksa otomatik mi yapayım?
4. **Eval seti** senin PDF'lerinden 30-50 soru gerektiriyor. Ben taslak üretip
   sana doğrulatmamı ister misin, yoksa soruları sen mi yazmak istersin?
   (Doğruluk metriklerinin anlamlı olması buna bağlı.)

---

## 5. Şu an dokunulmayacaklar

Bunlar çalışıyor ve bilinçli tasarlanmış — hiçbir fazda yeniden yazılmayacak:

- Tüm Foundry Local workaround'ları (`_patch_cli_compat`, `_ensure_model_loaded`,
  `capture_output` kullanmama kararı, `TRANSPORT_ERRORS` çift-kütüphane yakalama,
  `_create_chat_completion` fallback'i)
- `_ThinkStreamStripper`, `_RepetitionGuard`, `_stream_with_warmup` ve token
  bütçesi eskalasyonu
- `rrf_merge` (skor ölçeği sorununu doğru çözüyor)
- Türkçe "i" ailesi katlama (`_I_FOLD_MAP`) ve `_detect_language`'ın ucuz-kontrol-önce mantığı
- Özet map-reduce + kapasite hatasında bölerek yeniden deneme
- `db.close()` ile uzun LLM çağrısı öncesi session bırakma deseni
- `generate_answer`/`generate_quiz`/`generate_summary` non-stream sürümleri (testler için)
