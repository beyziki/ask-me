# Foundry Local: gözlemlenen sorunlar ve koddaki karşılıkları

Foundry Local henüz olgunlaşmamış bir yerel servis. Bu proje geliştirilirken
karşılaşılan davranışları ve her birine karşı koda eklenen korumayı burada
topluyoruz — hem ileride benzer bir hatayla karşılaşınca tanıyabilmek, hem de
koddaki "neden böyle yazılmış?" sorularının cevabı burada olsun diye.

Ana README'yi okunabilir tutmak için bu ayrıntılar oradan buraya taşındı.

---

## 1. Akıl yürütme (thinking) modu kapatılamıyor

**Sorun.** Qwen3 hibrit bir "thinking" modeli: cevaptan önce bir
`<think>...</think>` bloğu üretiyor. Bu blok kullanıcıya gösterilmiyor ama
token bütçesinden yiyor. Sonuç:

- her soru gereksiz yere yavaş;
- bazen tüm bütçe akıl yürütmede tükeniyor ve geriye görünür cevap kalmıyor —
  ekranda ya tamamen boş bir balon, ya da tek bir nokta (`"."`) kalıyordu.

**Denenenler.**

1. `/no_think` soft-switch'i (kullanıcı mesajının sonuna eklenir): yardımcı
   oluyor ama garanti değil.
2. Sistem prompt'una "`<think>` kullanma" talimatı: aynı şekilde zayıf bir
   sinyal.
3. Qwen3/vLLM ekosisteminin resmî yolu olan
   `chat_template_kwargs: {"enable_thinking": false}`: **Foundry Local'in REST
   API'si bu alanı hiç tanımıyor.** Kabul edilen alan listesinde yok
   ([REST API referansı](https://learn.microsoft.com/en-us/azure/foundry-local/reference/reference-rest)),
   ve `BadRequestError` de fırlatmıyor — sessizce yok sayılıyor. Yani istek
   modele hiç ulaşmıyor, ama biz de hatayı görmediğimiz için fallback'e
   düşmüyoruz. İlgili açık issue:
   [microsoft/foundry-local#808](https://github.com/microsoft/foundry-local/issues/808).
4. Qwen3'ün diğer resmî alternatifi (asistan turunu boş bir
   `<think>\n\n</think>` ile prefill edip `continue_final_message` kullanmak):
   Foundry Local'in mesaj şemasında karşılığı yok.

**Çözüm.** Varsayılan modeli **düşünmeyen (instruct)** bir modele çevirdik
(`mistral-nemo-12b-instruct`). Böyle bir modelde üretilen her token doğrudan görünür
cevap: hem çok daha hızlı, hem de boş-cevap arızası kökten ortadan kalkıyor.

Kod artık modele göre davranıyor (bkz. `config.py:model_has_thinking`):

| | Düşünmeyen model | Thinking model |
|---|---|---|
| `/no_think` eki | eklenmez | eklenir |
| "`<think>` kullanma" talimatı | eklenmez | eklenir |
| `enable_thinking=False` denemesi | yapılmaz | yapılır (destekleniyorsa işe yarar) |
| Token bütçesi | tek: 800 | iki kademeli: 1500 → 4096 |

`.env`'de `FOUNDRY_MODEL_ALIAS=qwen3-8b` yazman yeterli; kod tanıyıp sağdaki
sütuna geçiyor. `MODEL_THINKING=true/false` ile elle de zorlanabilir.

---

## 2. Model bozuk bir tekrar döngüsüne giriyor

**Gözlem.** Model (özellikle GPU'daki quantize sürümlerde) bazen bozuk bir
döngüye giriyor. Bunun İKİ ayrı biçimi gözlendi:

- **Tek kelime döngüsü.** Aynı kısa kelime onlarca-yüzlerce kez art arda:
  `"Otonom Otonom Otonom ... O O O O ..."`.
- **Öbek döngüsü.** Çok kelimeli bir öbek dönüp duruyor — bu biçimde AYNI
  kelime hiçbir zaman art arda gelmiyor:
  `"... çapkalarda kullanma ve kullanma için kullanma, çapkalarda kullanma ve
  kullanma için kullanma, ..."`.

  İkinci biçim uzun süre yakalanmadı: koruma yalnızca ardışık aynı kelimeyi
  sayıyordu, bu yüzden öbek döngüsünde hiç tetiklenmiyor ve kullanıcı ekranı
  dolduran anlamsız metnin tamamını görüyordu.

**Koruma (iki katmanlı).**

1. İsteklere `frequency_penalty` / `presence_penalty` ekleniyor — modele
   tekrarı caydırıcı bir sinyal.

   > **ÖLÇÜLDÜ (2026-08-27): bu katman ÇALIŞIYOR, ama tek başına yeterli
   > değil.** Aynı `random_seed` ile `frequency_penalty` 0.0 ve 2.0 farklı
   > metin üretti — yani parametre Foundry Local üzerinden sampler'a
   > gerçekten ulaşıyor. (Önceki tahmin, ONNX Runtime GenAI'nin `search`
   > şemasında bu alanların olmamasına dayanıyordu; Foundry Local arada bir
   > çeviri yapıyor olmalı.) Ölçüm: `evals/sampling_probe.py --determinism`.

2. Buna rağmen olursa `_RepetitionGuard` (bkz. `services/llm.py`) akışı
   `DegenerateOutput` ile erken kesiyor. Guard iki kuralı birlikte
   uyguluyor:
   - aynı kelime art arda 10+ kez (`max_consecutive_repeats`);
   - 2–12 kelimelik bir öbek 4 kez ardışık tekrar (`_MAX_PHRASE_PERIOD` /
     `_MIN_PHRASE_REPEATS`). Karşılaştırma büyük/küçük harf ve noktalama
     farkını yok sayıyor, çünkü döngüdeki öbek her turda birebir aynı
     yazılmıyor.

   Tek kelimenin tekrarı öbek kuralına takılmıyor (blok tek kelimeden
   ibaretse atlanıyor); yoksa `max_consecutive_repeats` eşiği sessizce 8'e
   inerdi.

   Kullanıcı yüzlerce token'lık anlamsız tekrar yerine anlaşılır bir hata
   görüyor (`/ask` için 503, `/ask/stream` için bir `error` olayı).

**Not.** Bu döngüler tipik olarak modelin dayanacak bir bağlamı olmadığında
(`has_context=False`) ortaya çıkıyor — yani asıl tetikleyici çoğu zaman
retrieval tarafındaki bir sorun oluyor (bkz. aşağıdaki "Seçili dosyada arama"
notu).

---

## 3. Boş / anlamsız derecede kısa cevap

Yukarıdaki 1. maddenin sonucu. Thinking modelinde hâlâ olabileceği için
korumalar duruyor:

- `_is_blank` — tamamen boş cevap.
- `_looks_too_short` — geriye yalnızca birkaç karakter kalmış "cevap"
  (gözlemlenen: tek bir `"."`). `_is_blank` bunu yakalamıyor çünkü metin
  teknik olarak boş değil.
- `_stream_with_warmup` — akışta, en az 4 görünür karakter birikene kadar
  çağırana hiçbir token yayınlanmaz. Eşiğe hiç ulaşılmazsa kullanıcı yarım bir
  metin görmemiş olur, bu yüzden sessizce yeni bir deneme başlatmak güvenlidir.
- Otomatik yeniden deneme, giderek genişleyen token bütçeleriyle
  (`config.py:answer_token_budgets`). Aynı bütçeyle tekrar denemenin faydası
  olmadığı gözlendiği için bu gerçek bir eskalasyon.
- Hepsi başarısız olursa `EmptyAnswer` — sessiz bir boşluk yerine açık bir
  hata mesajı.

Teşhisi kolaylaştırmak için, boş/kısa bir sonuçla karşılaşıldığında hangi
deneme numarasının hangi `max_tokens` bütçesiyle başarısız olduğu (ve `/ask`'te
`completion_tokens` / `finish_reason`) backend logunda `logger.warning` ile
görünüyor.

---

## 4. Bağlantı üretim biter bitmez yarıda kopuyor

**Gözlem.** `RemoteProtocolError: peer closed connection without sending
complete message body (incomplete chunked read)` — Foundry Local, üretim
bittikten hemen sonra chunked-encoding'in kapanış işaretini göndermeden
bağlantıyı kesebiliyor.

**Koruma** (`_stream_and_strip`): içerik zaten akmışsa akış sessizce
sonlandırılıyor (kullanıcı büyük ihtimalle cevabın tamamını almıştır); hiç
içerik gelmeden koparsa bu gerçek bir sorunun işareti, kullanıcıya anlaşılır
bir hata gösteriliyor.

---

## 5. Boş `choices` listesi taşıyan chunk

**Gözlem.** OpenAI'ın gerçek API'sinde `choices` yalnızca
`stream_options={"include_usage": true}` istendiğinde ve yalnızca sonda boş
gelir. Foundry Local bunu istenmeden de gönderebiliyor; `event.choices[0]`
doğrudan indekslenince `IndexError` ile çöküyordu.

**Koruma:** içerik taşımayan bu tür chunk'lar sessizce atlanıyor.

---

## 6. SDK / CLI uyumsuzlukları

- **`foundry service` → `foundry server`.** foundry-local-sdk 0.5.1, servis
  yönetimi için `foundry service ...` çağırıyor; CLI 0.10.3'te bu alt komut
  `foundry server ...` oldu. `_patch_cli_compat` (bkz. `services/llm.py`)
  çalışma zamanında, süreç içinde yamalıyor — site-packages'e dokunulmuyor.
- **Kırık katalog route'u.** SDK'nın `download_model` / `load_model` /
  `get_model_info` metodları, CLI 0.10.3'ün REST API'sinde artık olmayan
  `/foundry/list` route'unu kullanıyor (404). Bu yüzden manager
  `bootstrap=False` ile açılıp model yükleme ve model-id çözümlemesi elle
  yapılıyor (CLI `model load` + `/v1/models`).
- **Servis "hazır" görünüp bağlantı kabul etmiyor.** `foundry server status`
  çıktısında URI görünmesi, o adreste HTTP dinleyicisinin gerçekten hazır
  olduğu anlamına gelmiyor (soğuk başlangıçta WinError 10061). Bu yüzden metni
  bulduktan sonra ayrıca gerçek bir HTTP isteğiyle doğruluyoruz.
- **`capture_output=True` CLI'ı bozuyor.** `foundry model load` Python'dan
  `capture_output=True` ile (yani gerçek bir TTY'ye bağlı olmadan) çağrıldığında
  tutarlı biçimde "Daemon did not start listening within 15s" veriyor; tam aynı
  komut gerçek konsolda sorunsuz çalışıyor. Bu yüzden stdout/stderr **bilerek**
  pipe'a yönlendirilmiyor; alt süreç backend'in konsolunu miras alıyor.

---

## 7. "Daemon did not start listening within 15s" — takılı daemon / named pipe

**Kök neden (log ile doğrulandı).** Askıda kalmış bir `foundrylocald.exe`,
KULLANICIYA ÖZEL bir named pipe'ı tutmaya devam ediyor:

```
foundry-cli-S-1-5-21-...-<kullanıcı>
```

`foundry server status` "Not running" dese bile süreç ayakta olduğu için her
yeni daemon açılışta o pipe'ı alamayıp anında çıkıyor. `foundry server logs`
içinde net biçimde görünüyor:

```
Foundry.Daemon.Lifecycle.Contracts.DaemonBindContentionException:
    Another daemon already owns named pipe 'foundry-cli-S-1-5-21-...'
[WRN] IPC bind contention — another daemon already owns the per-user address;
    exiting with AlreadyRunning (75)
```

CLI bu anlık çıkışı "15 saniyede dinlemeye başlamadı" diye raporluyor —
yanıltıcı, çünkü daemon aslında hiç başlamıyor.

**YETKİ KURALI (en önemli madde).** Daemon YÜKSELTİLMİŞ (yönetici) bir
terminalde başlatılırsa iki şey birden bozulur:

1. Normal yetkiyle çalışan backend, yükseltilmiş daemon'ın named pipe'ına
   erişemez; erişemeyince KENDİ daemon'ını başlatmaya çalışır ve yukarıdaki
   bind contention'a düşer. `foundry model load`'un başarısız olmasının
   sebebi budur.
2. Yükseltilmiş süreç normal yetkiyle sonlandırılamaz ("Erişim engellendi"),
   yani temizlemek için her seferinde yönetici terminali gerekir.

Bu yüzden Foundry'i **her zaman backend ile aynı (normal) yetki seviyesinde**
çalıştır. Yönetici terminalini yalnızca takılı bir süreci öldürmek için kullan.

**Çözüm.** `scripts/foundry-doctor.ps1` bu döngüyü kırar: durumu kontrol eder,
takılı süreci temizler, servisi başlatır ve gerçekten hazır olduğunu doğrular.
Yönetici olarak çalıştırıldığında servisi başlatmayı reddeder (yukarıdaki
kural gereği), yalnızca temizlik yapar.

```powershell
.\scripts\foundry-doctor.ps1              # normal terminal: temizle + başlat
.\scripts\foundry-doctor.ps1 -CleanOnly   # yönetici terminal: sadece temizle
```

### Elle müdahale

`foundry model load`, servis Python tarafından doğrulanmış şekilde çalışıyor
olsa bile bazen bunu göremeyip kendi daemon başlatma denemesini yapıyor ve
kendi 15 saniyelik iç zaman aşımına takılıyor. Python tarafından önlenemiyor,
bu yüzden birkaç kez tekrar deniyoruz.

Bu hatayı görürsen:

1. backend'i durdur,
2. `foundry server status` ile servisi kontrol et,
3. Görev Yöneticisi'nde takılı kalmış bir `foundrylocald.exe` varsa sonlandır,
4. ayrı bir terminalde `foundry server start`'ın temiz şekilde ayağa
   kalktığını doğrula,
5. backend'i tekrar başlat.

**Sık karşılaşılan tetikleyici:** `uvicorn --reload`'u `--reload-dir backend`
olmadan çalıştırmak. O durumda `data/ask_me.db` ve `data/vectorstore/*.faiss`
her istekte değiştiği için backend süreci soru sorulurken yeniden başlıyor;
tam da Foundry Local ile konuşurken öldürülen bir süreç, bu hataya ve bazen
daemon'ın tamamen takılmasına yol açıyor. Bkz. README'deki çalıştırma komutu.

---

## "Bağlantı kesildi" aslında VRAM yetmemesi

**Gözlem.** Arayüzde şu hata çıkıyor:

> Foundry Local ile bağlantı üretim başlamadan/sırasında beklenmedik şekilde
> kesildi.

Backend tarafında asıl istisna `RemoteProtocolError: peer closed connection
without sending complete message body`. Foundry `200 OK` dönüyor, model
"loaded successfully" diyor, retrieval de sorunsuz çalışıyor
(`en iyi skor=0.526 -> has_context=True`) — dışarıdan hiçbir şey yanlış
görünmüyor.

**Gerçek sebep** yalnızca `foundry server logs` içinde görünüyor:

```
OnnxRuntimeGenAIException: CUDA error in CudaMallocArray
  at ...\onnxruntime-genai\src\cuda\cuda_common.h:131 - out of memory
  at Microsoft.ML.OnnxRuntimeGenAI.Generator..ctor(Model, GeneratorParams)
```

Kritik ayrıntı yığındaki konum: çökme `Generator` **kurulurken** oluyor.
Yani modelin ağırlıkları GPU'ya çoktan yerleşmiş; yer kalmayan şey o isteğin
**KV cache**'i. Bu yüzden "model başarıyla yüklendi" satırı yanıltıcı — model
gerçekten yüklü, üretemeyen şey istek.

**Neden bu kadar geç anlaşıldı.** Üç katman birden sebebi gizliyordu:

1. Foundry, hata çıktığında bile `200 OK` + `text/event-stream` gönderiyor
   (başlıklar gövdeden önce yazılıyor), sonra bağlantıyı kapatıyor. HTTP
   seviyesinde istek başarılı görünüyor.
2. `api/ask.py` bu istisnayı yakalayıp kullanıcıya mesaj gösteriyor ama
   `__cause__`'u hiç loglamıyordu — geliştirici boş terminalle kalıyordu.
   (Düzeltildi: artık `asil sebep: ...` diye yazılıyor.)
3. Uygulama logları zaten görünmüyordu; uvicorn yalnızca kendi logger'larını
   yapılandırıyor ve kök logger'da handler olmadığı için `logger.info`
   sessizce kayboluyordu. (Düzeltildi: `main.py`'de `logging.basicConfig`.)

**Ne yapmalı.**

- **Önce GPU'yu boşalt.** `nvidia-smi` çıktısındaki süreç listesine bak. 8 GB'lık
  bir dizüstü kartında arka planda duran bir oyun ya da launcher (gözlemlenen:
  `EpicGamesLauncher.exe` ve bir Unreal Engine oyun süreci) tek başına
  gigabaytlarca VRAM tutabiliyor. Bunları kapatmak çoğu zaman yeterli.
  > `nvidia-smi`'yi model YÜKLÜYKEN çalıştır. Foundry modeli TTL sonunda
  > boşaltıyor (daemon log'unda arka arkaya "loaded successfully" satırları
  > bunun izi); boşken ölçüm alırsan kart bomboş görünür ve yanıltır.
- **KV cache'i küçült.** `.env`'de `ANSWER_MAX_TOKENS`, `MAX_CONTEXT_CHUNKS`
  ve `MAX_CONTEXT_CHARS` değerlerini düşür — KV cache boyutu (bağlam +
  üretilecek token) uzunluğuyla doğru orantılı.
- **Embedding modelini GPU'dan uzak tut** (varsayılan artık öyle; bkz. bir
  sonraki bölüm).

---

## Embedding modeli GPU'yu dil modelinden çalıyor

**Gözlem.** `evals/sampling_probe.py` prompt önbelleği boşken çalıştırıldığında
Foundry Local'e giden HER istek 500 döndü:

```
CUDA error in CudaMallocArray at ...cuda_common.h:131 - out of memory
```

Aynı araç, prompt önbellekten geldiğinde (yani embedding modeli hiç
yüklenmediğinde) sorunsuz çalışıyordu. Fark buydu.

**Sebep.** `sentence-transformers`, `device` verilmediğinde CUDA varsa
otomatik GPU'ya yerleşiyor. Ama GPU'da zaten Foundry Local'in dil modeli
oturuyor (3.6 GB + KV cache) ve kart 8 GB. İkisi aynı anda yer istediğinde
ONNX Runtime GenAI ayırma yapamıyor.

**Bu YALNIZCA probe'un sorunu değil — üretimde de aynı şey oluyor.** Backend
de aynı `get_embedder()`'ı çağırıyor, yani `uvicorn` süreci de embedding
modelini GPU'ya koyuyordu. Kartta yer kaldığı sürece görünmüyor; kalmadığında
`GpuContextLost` olarak patlıyor (bkz. yukarıdaki ilgili bölüm).

**Düzeltme.** `EMBEDDING_DEVICE` ayarı eklendi, varsayılanı `cpu`
(bkz. `core/config.py`). Embedding modeli küçük ve soru başına tek bir kısa
metin gömüyor — bu iş CPU'da milisaniyeler sürüyor, karşılığında VRAM'in
tamamı dil modeline kalıyor.

**Takas.** Dosya yüklemedeki toplu gömme CPU'da daha yavaş. Rahatsız ederse
ölç (`python -m benchmarks.bench_ingestion`) ve kartında yer varsa `.env`'de
`EMBEDDING_DEVICE=cuda` yap.

> Not: mevcut FAISS vektörleri GPU'da hesaplanmıştı, yeni sorgular CPU'da
> hesaplanacak. Aradaki kayan nokta farkı ~1e-6 mertebesinde; kosinüs
> sıralamasını etkilemiyor, yeniden indeksleme gerekmiyor.

---

## `temperature`, `top_p` ve `top_k` YOK SAYILIYOR — üretim greedy

**Gözlem.** `evals/sampling_probe.py` ile aynı prompt üzerinde beş ayrı
örnekleme ayarı denendi:

| ayar | çıktı |
|---|---|
| `temperature=0.2` | 2435 karakter |
| `temperature=0.2, top_p=0.9` | 2435 karakter |
| `temperature=0.7, top_p=0.9` | 2435 karakter |
| `temperature=0.7, top_p=0.9, top_k=40` | 2435 karakter |
| `temperature=1.0, top_p=0.95` | 2435 karakter |

Beşi de **birebir aynı metni** üretti (aynı karakter sayısı, aynı hash).
`temperature=0.2` ile `temperature=1.0` arasında tek karakter fark yok.

**Sebep.** ONNX Runtime GenAI'nin `do_sample` varsayılanı `false`. O haldeyken
motor **greedy search** yapıyor: her adımda en yüksek olasılıklı token
seçiliyor ve `temperature` / `top_p` / `top_k` hiç devreye girmiyor. Foundry
Local'in REST şeması bu üç alanı kabul ediyor, ama `do_sample`'ı açmanın bir
yolu sunmuyor.

**Üç sonucu var.**

1. **Üretim deterministik.** Aynı prompt her zaman aynı cevabı veriyor.
   Kodun bazı yerlerinde "örnekleme deterministik değil, tekrar denemek işe
   yarar" varsayımı vardı (bkz. `llm.py:_answer_token_budgets` üzerindeki
   not) — bu varsayım YANLIŞ. Aynı prompt'u aynı parametrelerle yeniden
   denemek birebir aynı sonucu verir; boş/bozuk cevap için yeniden deneme
   ancak `max_tokens` gibi bir şey DEĞİŞİYORSA anlamlı.

2. **Tekrar döngüsü rastlantısal değil.** Belirli prompt'larda her zaman
   oluyor, diğerlerinde hiç olmuyor. Bu yüzden `sampling_probe` aynı soruyu
   N kez değil, N FARKLI SORUYU bir kez çalıştırıyor — aynı soruyu
   tekrarlamak deterministik bir sistemde hiçbir bilgi üretmiyor.

3. **Greedy çözümleme döngünün ta kendisi.** Tekrar döngüsü greedy
   çözümlemenin bilinen arızasıdır (Holtzman ve ark., 2019): model bir kez
   tekrara girdiğinde, bağlam artık aynı token'ı en yüksek olasılıkla
   öngörür ve olasılıksal bir kaçış yolu olmadığı için döngüden çıkamaz.
   Sampling açık olsaydı düşük olasılıklı bir token eninde sonunda zinciri
   kırardı.

**Elde kalan kaldıraçlar.** `temperature`/`top_p`/`top_k` ayarlamak boşuna;
geriye üç şey kalıyor: (a) `frequency_penalty`/`presence_penalty` — greedy
çözümlemede de çalışıyor çünkü doğrudan logit'leri değiştiriyor;
(b) prompt tarafında cevabı kısaltmak; (c) `_RepetitionGuard` (son çare).

---

## Seçili dosyada arama boş dönüyordu (retrieval)

**Gözlem.** Kullanıcı aramayı tek bir dosyayla sınırladığında ("Yalnızca
seçili 1 dosyada aranacak") sistem neredeyse her soruda *"Soru, yüklenen
dosyalarda bir bölüm bulunamadı"* diyor ve modeli bağlamsız bırakıyordu —
dosya soruyla birebir ilgili olsa bile.

**Sebep.** `services/rag.py:semantic_search`, FAISS'e **global** bir arama
yapıp (`index.search(q, top_k)`) dönen satırlardan kapsam dışı olanları
sonradan atıyordu. Seçili dosyanın parçaları tüm korpusun genel
sıralamasında ilk `top_k` (8) içine giremezse geriye hiçbir şey kalmıyordu.
1025 chunk'lık gerçek bir index üzerinde ölçüldü: 11 chunk'lık tek bir dosya
seçiliyken sorguların **%90'ında** semantic arama boş dönüyordu. Boş sonuç
`api/ask.py:_retrieve`'de `best_semantic = 0.0` demek, o da `has_context =
False` demek.

**Düzeltme.** Arama artık kapsamın İÇİNDE yapılıyor: izin verilen satırların
vektörleri `reconstruct_batch` ile index'ten geri okunup doğrudan iç çarpımla
puanlanıyor. Vektörler normalize olduğu için skor ölçeği global aramayla
birebir aynı (ölçülen sapma < 1e-7). Kapsam tüm index olduğunda FAISS'in
kendi (optimize) araması kullanılmaya devam ediyor.

**Teşhis.** `_retrieve` artık her soruda kapsam boyutunu, aday sayısını ve en
iyi semantic skoru loglıyor. Backend terminalinde:

```
retrieval: kapsam=11 chunk, semantic aday=8, en iyi skor=0.412, eşik=0.30 -> has_context=True
```

Hâlâ `has_context=False` görüyorsan skora bak: eşiğe yakınsa `.env`'de
`MIN_RELEVANCE_SCORE`'u düşür, çok düşükse soru gerçekten o dosyayla ilgili
değil demektir.
