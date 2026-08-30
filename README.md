# Ask Me? — Offline AI Study Assistant

Bilgisayar mühendisliği öğrencileri için **tamamen çevrimdışı** çalışan, çok
kullanıcılı bir ders çalışma asistanı. Kendi ders notlarını (PDF, Markdown, kod
dosyası) yükleyip üzerine soru sorabilir, otomatik quiz üretebilir ve kod
dosyalarını açıklattırabilirsin. Hiçbir veri makineden dışarı çıkmaz: hem dil
modeli hem de arama indeksleri yerelde çalışır.

---

## Ne yapıyor?

| Özellik | Açıklama |
|---|---|
| **Doküman yükleme** | PDF / Markdown / kod dosyası → metin çıkarma → chunking → indeksleme |
| **Hybrid RAG** | Semantic search (FAISS) + BM25, Reciprocal Rank Fusion ile birleştirilir |
| **Kaynak gösterimli cevap** | Cevap, kullanılan doküman parçalarıyla birlikte gösterilir |
| **Canlı akış (streaming)** | Cevap token token yazılır (SSE) |
| **Özet çıkarma** | Bir dosyanın tamamını özetler (uzun dosyalarda map-reduce), özeti saklar |
| **Otomatik quiz** | Çoktan seçmeli / açık uçlu sorular üretir — özet varsa ondan (çok daha hızlı) |
| **Kod analizi** | Yüklenen kod dosyasını açıklar |
| **Dosya gruplama** | Dosyalar derse/konuya göre gruplanıp aramada filtrelenebilir |
| **Çok kullanıcılı** | Her kullanıcının dokümanları, indeksi ve geçmişi izole |
| **Türkçe / İngilizce** | Soru dili otomatik algılanır, cevap aynı dilde verilir |

---

## Mimari

```
React + Vite (frontend-web, :5173)
        │  HTTP / SSE
        ▼
FastAPI (backend, :8000)
        ├── SQLite (kullanıcılar, dokümanlar, chunk'lar, quiz'ler)
        ├── FAISS + BM25  ← Hybrid RAG
        └── Foundry Local ← yerel LLM
```

**Backend neden Docker'da değil?** Foundry Local ile Windows'a özel `foundry`
CLI'ı üzerinden konuşuyor (bkz. `backend/app/services/llm.py`), bu yüzden
native Windows'ta çalıştırılmalı. Frontend container'da çalışabilir; backend'e
her zaman tarayıcıdan doğrudan bağlanır.

### Teknoloji yığını

- **Backend:** FastAPI, SQLAlchemy + SQLite, pydantic-settings
- **Frontend:** React + TypeScript + Vite + TailwindCSS
- **RAG:** sentence-transformers (embedding), FAISS (vektör arama), rank_bm25
- **LLM:** Microsoft Foundry Local (yerel, çevrimdışı model çalıştırma)
- **Dosya işleme:** pypdf
- **Test:** pytest

---

## Kurulum

### 1) Foundry Local ve model

```powershell
foundry model list                          # kullanılabilir modeller
foundry model download ministral-3-3b-instruct-2512
foundry model load ministral-3-3b-instruct-2512
```

> **Model seçimi önemli.** Varsayılan olarak **düşünmeyen (instruct)** bir model
> kullanılıyor. Qwen3 gibi "thinking" modelleri cevaptan önce görünmeyen bir
> akıl yürütme bloğu üretir; Foundry Local bunu kapatmanın bir yolunu sunmuyor,
> bu yüzden her soru hem çok daha yavaş oluyor hem de bazen tüm token bütçesi
> akıl yürütmede tükenip boş cevap dönüyordu. Ayrıntı:
> [`docs/foundry-local-notlari.md`](docs/foundry-local-notlari.md).
>
> Qwen3'e dönmek istersen tek yapman gereken `.env`'de alias'ı değiştirmek —
> kod modeli tanıyıp `/no_think` ve geniş token bütçesi davranışına otomatik
> geçiyor.

### 2) Backend

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3) Frontend

```powershell
cd frontend-web
npm install
```

---

## Çalıştırma

**Backend** (native çalışmalı):

```powershell
uvicorn backend.app.main:app --reload --reload-dir backend
```

> `--reload-dir backend` gerekli: yalnız `--reload`, proje kökündeki her şeyi
> (`.venv/`, `node_modules/` ve her istekte değişen `data/ask_me.db`,
> `data/vectorstore/*.faiss`) izleyip her soruda backend'i yeniden başlatır.

**Frontend** — Docker ile (önerilen):

```powershell
docker compose up -d --build
```

veya doğrudan:

```powershell
cd frontend-web
npm run dev
```

Her iki durumda arayüz `http://localhost:5173`, backend `http://127.0.0.1:8000`.

Hangi modelin gerçekten kullanıldığını doğrulamak için:
`http://127.0.0.1:8000/health/model`

---

## Yapılandırma

Tüm ayarlar `backend/app/core/config.py` içinde ve proje kökündeki `.env`
dosyasıyla değiştirilebilir. En sık kullanılanlar:

| Ayar | Varsayılan | Ne işe yarar |
|---|---|---|
| `FOUNDRY_MODEL_ALIAS` | `ministral-3-3b-instruct-2512` | Kullanılacak Foundry Local modeli |
| `MODEL_THINKING` | (alias'tan çıkarılır) | Modelin akıl yürütme modu var mı |
| `ANSWER_MAX_TOKENS` | `800` | Cevap uzunluğu üst sınırı |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `220` / `40` | Chunking (kelime cinsinden) |
| `TOP_K_SEMANTIC` / `TOP_K_BM25` | `8` / `8` | Her aramadan alınan aday sayısı |
| `MAX_CONTEXT_CHUNKS` | `5` | LLM'e gerçekten gönderilen parça sayısı |
| `SUMMARY_GROUP_MAX_CHARS` | `8000` | Özetlemede bir map grubunun boyut sınırı |
| `SUMMARY_MAX_TOKENS` | `1200` | Nihai özetin uzunluk sınırı |
| `MIN_RELEVANCE_SCORE` | `0.30` | Bunun altında "dosyalarda ilgili bilgi yok" sayılır |
| `WARMUP_ON_STARTUP` | `true` | Model/embedding'i açılışta arka planda ısıt |

`CHUNK_SIZE`'ı değiştirirsen mevcut dosyalar eski boyutta kalır — yeni ayarın
etkili olması için o dosyaları silip yeniden yüklemen gerekir.

---

## Kalite ve hız için tasarım kararları

"Neden böyle yazılmış?" sorularının cevabı:

**Arama (retrieval)**

- **Ortak tokenizasyon.** BM25 hem korpusu hem sorguyu aynı şekilde tokenize
  eder: küçük harfe indirir, noktalamayı atar ve dört Türkçe `i` harfini
  (`i ı İ I`) tek forma katlar. Böylece hem "İŞLEMCİ"/"işlemci" hem de
  "TURING"/"Turing" eşleşir. (Ham `split()` ile "Turing?" hiçbir zaman
  "Turing" ile eşleşmiyordu.)
- **Reciprocal Rank Fusion.** Semantic ve BM25 skorları farklı ölçeklerde
  (biri kosinüs benzerliği, diğeri kendi içinde normalize edilmiş bir sayı),
  bu yüzden toplanamaz. RRF yalnızca sıraya bakar ve iki listede de üstte
  çıkan parçaları öne alır.
- **Alaka eşiği.** En iyi semantic benzerlik `MIN_RELEVANCE_SCORE` altındaysa,
  bulunan parçalar yalnızca "en az alakasız" olanlardır. Bu durumda context
  hiç gönderilmez; model genel bilgisiyle cevap verir ve arayüz bunu
  kullanıcıya açıkça söyler.
- **İçerik bazlı dedupe.** Aynı dosya iki kez yüklendiyse `chunk_id`'ler farklı
  olur ama metin aynıdır; kırpmadan önce içerik bazında tekilleştiriyoruz.

**Hız**

- BM25 ve FAISS indeksleri süreç içinde önbelleklenir; eskiden her soruda
  sıfırdan kuruluyor / diskten okunuyordu.
- Chunk boyutu ve context üst sınırı, prefill süresini makul tutacak şekilde
  ayarlandı.
- Embedding modeli ve Foundry Local backend açılırken arka planda ısıtılır —
  soğuk başlangıç maliyeti ilk sorunun içinde ödenmez.
- Düşünmeyen bir model + tek ve küçük token bütçesi.

**Özet ve quiz hızı**

- **Map-reduce özetleme.** Özet dokümanın TAMAMINI kapsamak zorunda, ama uzun
  bir ders notu tek istekte modele sığmaz. Doküman ardışık gruplara bölünüp
  her grup ayrı özetleniyor (map), sonra ara özetler tek bir nihai özete
  indirgeniyor (reduce). Kısa dosyalarda map adımı hiç çalışmıyor — gereksiz
  ikinci tur maliyeti ödenmiyor. Gruplama sırayı koruyor: parçalar doküman
  sırasından çıkarsa anlatım bozuluyor.
- **Quiz özetten üretiliyor.** Özet varsa quiz varsayılan olarak ondan
  üretiliyor. İki kazanç birden: özet zaten damıtılmış ve kısa olduğu için
  prefill süresi düşüyor (belirgin hız artışı), ve özet dokümanın tamamından
  geldiği için kapsam, ham parçalardan yapılan örneklemeden daha iyi.
  Özet yoksa eski davranışa (parça örnekleme) düşülüyor.
- **Özetler kalıcı.** Doküman başına tek özet saklanıyor; tekrar açıldığında
  anında geliyor ve quiz üretimi de bekletmiyor.

**Sağlamlık.** Yerel modelin gözlemlenen arıza modlarına karşı korumalar
(bozuk tekrar döngüsü, boş cevap, yarıda kopan bağlantı) ve Foundry Local'e
özgü uyumluluk yamaları:
[`docs/foundry-local-notlari.md`](docs/foundry-local-notlari.md).

---

## Test

```powershell
pip install -r requirements.txt
python -m pytest tests/ -v
```

- **Ağ/Foundry gerektirmeyen birim testleri** (`test_llm_utils.py`,
  `test_ingestion.py`, `test_rag_merge.py`, `test_ask_retrieval.py`,
  `test_quiz_utils.py`): saf fonksiyonları test eder, her ortamda hızlı çalışır.
- **Entegrasyon testleri** (`test_multiuser.py`, `test_document_groups.py`):
  gerçek FastAPI uygulamasını geçici bir SQLite/vectorstore ile ayağa kaldırıp
  çok kullanıcılı izolasyonu ve doküman gruplama uçlarını uçtan uca doğrular.
  Dosya yükleme adımı, embedding modelinin yerelde önbellekte olmasını
  gerektirir (backend'i en az bir kez çalıştırdıysan zaten öyledir).

---

## API uçları

| Uç | Açıklama |
|---|---|
| `POST /users/register`, `POST /users/login` | Kullanıcı işlemleri |
| `POST /documents/upload` | Dosya yükle (isteğe bağlı `group_id`) |
| `GET /documents`, `DELETE /documents/{id}` | Listele / sil |
| `POST /documents/groups`, `GET`, `DELETE /{id}` | Grup işlemleri |
| `PATCH /documents/{id}/group` | Dokümanı gruba ata / gruptan çıkar |
| `POST /ask` | Tek seferde cevap (JSON) |
| `POST /ask/stream` | Cevabı token token akıt (SSE) |
| `POST /summary/{id}/stream` | Dokümanın tamamını özetle (SSE), sonucu kaydet |
| `GET /summary`, `GET /summary/{id}`, `DELETE /summary/{id}` | Özet durumu / getir / sil |
| `POST /quiz`, `POST /quiz/stream` | Quiz üret (`source`: `auto` / `summary` / `chunks`) |
| `POST /code/explain` | Kod dosyasını açıkla |
| `GET /health`, `GET /health/model` | Sağlık / model bilgisi |

`/ask/stream` olay sırası: `sources` → (birçok) `token` → `done`; hata olursa
`error`. `sources` olayı `has_context` alanını da taşır — false ise cevap
dosyalardan değil, modelin genel bilgisinden geliyor.

---

## Klasör yapısı

```
ask-me/
├── backend/app/
│   ├── api/          # FastAPI route'ları (users, documents, ask, quiz, code)
│   ├── core/         # config, güvenlik (argon2)
│   ├── db/           # SQLAlchemy modelleri ve session
│   ├── models/       # Pydantic şemaları
│   └── services/     # ingestion, rag, llm, quiz, code-analysis
├── frontend-web/     # React + Vite + Tailwind arayüzü (aktif)
├── frontend/         # eski Streamlit prototipi (kullanılmıyor)
├── docs/             # Foundry Local notları ve sorun giderme
├── scripts/          # foundry-doctor.ps1 (servis tanı/onarım)
├── data/
│   ├── uploads/      # kullanıcı bazlı yüklenen dosyalar
│   └── vectorstore/  # kullanıcı bazlı FAISS indeksleri
├── docker-compose.yml
└── tests/
```

> Proje başlangıçta Streamlit arayüzüyle planlanmıştı; daha esnek bir deneyim
> için React + Vite + Tailwind'e geçildi. Eski arayüz
> (`frontend/streamlit_app.py`) referans olarak duruyor ama kullanılmıyor.

---

## Sorun giderme

Foundry Local servisi takılırsa (`Daemon did not start listening within 15s`)
tanı/onarım betiği:

```powershell
.\scripts\foundry-doctor.ps1
```

Takılı daemon'ı temizler ve servisi doğru yetki seviyesinde başlatır.
**Önemli:** Foundry'i backend ile aynı (normal) yetki seviyesinde çalıştır —
yönetici terminalinde başlatılan bir daemon'a backend erişemez.

Diğer Foundry Local'e özgü hatalar (model yüklenmiyor, bağlantı yarıda
kopuyor, model boş cevap veriyor) ve bunlara karşı koddaki korumalar:
[`docs/foundry-local-notlari.md`](docs/foundry-local-notlari.md)

## Final Teslim

**Final Sunumu:** [2 dakikalık sunumu izlemek için tıklayın] (https://drive.google.com/file/d/1e-Rot8WyLyb5MmcdHX6qFRfic9Ft8bby/view?usp=sharing)
