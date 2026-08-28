# Ask Me? — Çalıştırma Kılavuzu

Terminalde kaybolduğunda buraya bak. Her komut PowerShell içindir.

---

## 0. Dört altın kural

Şu ana kadarki hataların hepsi bu dördünden birine denk geldi.

1. **Her zaman `ask-me` dizininde ol.** Bir üstte (`Microsoft-Internship`) olursan
   `ModuleNotFoundError: No module named 'backend'` alırsın. `docker compose` de
   "no configuration file provided" der.
2. **Her zaman venv aktif olsun.** Prompt'un başında `(.venv)` yazmalı. Yazmıyorsa
   sistem Python'ı çalışıyordur ve orada `faiss`/`sentence-transformers` yok.
3. **Komutları tek tek çalıştır.** Çok satırlı blok yapıştırma — PowerShell kalan
   satırları çalışan komuta girdi olarak besliyor.
4. **Ctrl+C basma.** `torch` + `sentence-transformers` + `faiss` ilk import'ta
   Windows'ta 20-40 saniye sürebiliyor (Defender site-packages'ı tarıyor).
   Takıldı sanma; sessizlik normal.

Her terminal oturumunun başlangıcı **her zaman** şu iki satır:

```powershell
cd C:\Users\monster\Documents\Microsoft-Internship\ask-me
.\.venv\Scripts\Activate.ps1
```

Doğru yerde olduğunu bir kez teyit et:

```powershell
python -c "import sys; print(sys.executable)"
```

Çıktı `...\ask-me\.venv\Scripts\python.exe` olmalı. Değilse venv aktif değildir.

> `Activate.ps1` "script çalıştırma engellendi" derse, o oturum için izin ver:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> ```

---

## 1. Projeyi açıp kullanmak

Üç terminal gerekiyor. Sırayla.

### Terminal A — Foundry Local (model)

```powershell
foundry server start
foundry model load ministral-3-3b-instruct-2512
```

Bu terminali kapatabilirsin; servis arka planda kalır. Bir şeyler ters giderse:

```powershell
foundry server status
.\scripts\foundry-doctor.ps1
```

### Terminal B — Backend

```powershell
cd C:\Users\monster\Documents\Microsoft-Internship\ask-me
.\.venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --reload --reload-dir backend
```

`--reload-dir backend` **önemli**: yalnız `--reload` yazarsan uvicorn proje
kökündeki her şeyi izler — `.venv/`, `node_modules/` ve her istekte değişen
`data/ask_me.db`, `data/vectorstore/*.faiss` dahil — ve her soruda backend'i
yeniden başlatır (yani her soruda model baştan ısınır).

Beklenen çıktı:

```
INFO:     Application startup complete.
```

Model ısınması arka planda sürer, birkaç saniye sonra logda görürsün.

### Terminal C — Frontend

```powershell
cd C:\Users\monster\Documents\Microsoft-Internship\ask-me\frontend-web
npm run dev
```

Sonra tarayıcıda: **http://localhost:5173**

Backend `http://127.0.0.1:8000` üzerinde. Hangi modelin gerçekten yüklü
olduğunu görmek için: **http://127.0.0.1:8000/health/model**

### İlk kurulumsa (bir kereye mahsus)

```powershell
cd C:\Users\monster\Documents\Microsoft-Internship\ask-me
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd frontend-web
npm install
cd ..
```

---

## 2. Baseline ölçümü — **bir kereye mahsus, Faz 2'den önce**

Bu, projeyi kullanmak için gerekli değil. Optimizasyonların işe yarayıp
yaramadığını ölçebilmek için gereken "önce" fotoğrafı.

Backend'i **Terminal B'de açık bırak**, yeni bir terminal aç:

```powershell
cd C:\Users\monster\Documents\Microsoft-Internship\ask-me
.\.venv\Scripts\Activate.ps1
```

Sonra sırayla, **her birinin bitmesini bekleyerek**:

**1) Testler**

```powershell
pytest
```

Beklenen: `125 passed`. Farklıysa çıktıyı olduğu gibi paylaş.

**2) Performans benchmark'ları** (uzun sürer — `npm run build` ve soğuk model
yüklemesi dahil, 5-15 dakika)

```powershell
python -m benchmarks.run_all --label baseline
```

**3) RAG doğruluk ölçümü** (saniyeler)

```powershell
python -m evals.run_eval --label baseline
```

**4) Eşzamanlılık** (backend ayakta olmalı)

```powershell
python -m benchmarks.bench_concurrent --user-id 1
```

Sonunda şu iki dosya oluşur — Faz 2'den itibaren her değişiklik bunlara karşı
ölçülecek:

```
benchmarks\results\baseline.json
evals\results\baseline.json
```

### Ayrıca: eval setini gözden geçir

`evals\dataset.json` — 40 soruluk bir **taslak**. Yanlış bir "doğru cevap"
ondan sonraki her ölçümü sessizce bozar. Bakman gerekenler:

- Soru anlamlı mı?
- `expected_documents` doğru dosyayı gösteriyor mu?
- `required_terms` gerçekten o dokümana özgü mü?

Yanlış bulduğunu sil veya düzelt. Her vakanın `note` alanında neyi ölçtüğü yazıyor.

---

## 3. Sorun giderme

| Hata | Sebep | Çözüm |
|---|---|---|
| `ModuleNotFoundError: No module named 'backend'` | Yanlış dizindesin | `cd ...\ask-me` |
| Traceback'te `AppData\...\Python311\Lib\site-packages` geçiyor | venv aktif değil | `.\.venv\Scripts\Activate.ps1` |
| `KeyboardInterrupt`, `no tests ran` | Ctrl+C bastın veya blok yapıştırdın | Tek komut çalıştır, bekle |
| `[Errno 10048] ... bind on address ('127.0.0.1', 8000)` | Port dolu | Aşağı bak |
| `no configuration file provided` (docker) | Yanlış dizin | `cd ...\ask-me` |
| `Activate.ps1 ... engellendi` | ExecutionPolicy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` |
| Cevap gelmiyor / `FoundryNotAvailable` | Foundry kapalı | `foundry server start` |

### Port 8000 dolu

Kimin tuttuğuna bak:

```powershell
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000 -State Listen).OwningProcess
```

**`python` / `uvicorn` çıkarsa** — eski bir backend açık kalmış:

```powershell
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000 -State Listen).OwningProcess
```

**`com.docker.backend` / `wslrelay` çıkarsa** — bir Docker container'ı portu
yayınlıyor. Bu süreçleri öldürme, Docker Desktop'ın kendisi onlar. Önce hangi
container olduğuna bak:

```powershell
docker ps
```

Eski bir ask-me container'ıysa kaldır (kodun eski bir kopyasını çalıştırıyor,
ölçümleri bozar):

```powershell
docker compose down
```

Alakasızsa portu değiştir:

```powershell
uvicorn backend.app.main:app --port 8001 --reload --reload-dir backend
```

ve frontend'in nereye bakacağını da güncelle — `frontend-web\.env.development`:

```
VITE_API_BASE_URL=http://127.0.0.1:8001
```

Benchmark için de:

```powershell
python -m benchmarks.bench_concurrent --user-id 1 --base-url http://127.0.0.1:8001
```

---

## 4. Faz 2'den sonraki döngü

Ben bir optimizasyon yapıp dosyaları diskine yazdığımda, doğrulaman şu üç
komut. Sırayla:

```powershell
pytest
```

```powershell
python -m benchmarks.run_all --label faz2
python -m benchmarks.compare baseline faz2
```

```powershell
python -m evals.run_eval --label faz2
python -m evals.compare baseline faz2
```

`compare` komutları:

- **hızlanmayı** `↓` ve yeşil `İYİ` ile,
- **yavaşlamayı / doğruluk kaybını** `↑` ve `KÖTÜ` ile,
- ve eval tarafında **hangi sorunun bozulduğunu tek tek** gösterir.

Herhangi bir regresyonda çıkış kodu 1 dönerler. Bozulan bir şey varsa değişikliği
geri alırız — çıktıyı olduğu gibi paylaş yeter.

---

## 5. Tek bakışta komut listesi

```powershell
# --- her oturumun başı ---
cd C:\Users\monster\Documents\Microsoft-Internship\ask-me
.\.venv\Scripts\Activate.ps1

# --- projeyi kullanmak (3 ayrı terminal) ---
foundry server start                                        # A
foundry model load ministral-3-3b-instruct-2512             # A
uvicorn backend.app.main:app --reload --reload-dir backend  # B
cd frontend-web ; npm run dev                               # C
# tarayıcı: http://localhost:5173

# --- baseline (bir kere) ---
pytest
python -m benchmarks.run_all --label baseline
python -m evals.run_eval --label baseline
python -m benchmarks.bench_concurrent --user-id 1

# --- her değişiklikten sonra ---
pytest
python -m benchmarks.run_all --label <faz> ; python -m benchmarks.compare baseline <faz>
python -m evals.run_eval --label <faz>     ; python -m evals.compare baseline <faz>
```
