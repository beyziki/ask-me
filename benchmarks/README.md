# Benchmarks

Bu klasör **hiçbir üretim kodunu değiştirmez.** Yalnızca `backend.app...`
modüllerini olduğu gibi çağırıp ölçer.

Amaç tek bir kuralı mümkün kılmak: **önce ölç, sonra değiştir, tekrar ölç.**
"Daha hızlı hissettiriyor" bir kanıt değil.

## Hızlı başlangıç

```bash
# Çalıştırılabilen her şeyi koş, sonucu "baseline" olarak sakla
python -m benchmarks.run_all --label baseline

# ... bir optimizasyon yap ...

python -m benchmarks.run_all --label faz2
python -m benchmarks.compare baseline faz2
```

`compare`, bir metrik %5'ten fazla kötüleşirse **çıkış kodu 1** döner —
CI'da regresyonu kırmızıya çevirmek için.

## Suite'ler

| Suite | Ne ölçüyor | Gereksinim |
|---|---|---|
| `retrieval` | DB sorgusu, BM25 tokenizasyon/kurulum/arama, FAISS okuma/arama, sorgu embedding'i | `data/ask_me.db` + embedding modeli |
| `ingestion` | PDF parse, chunking, embedding, FAISS yazma (dosya boyutuna göre) | `data/uploads/` altında PDF + embedding modeli |
| `resources` | Kademeli RAM, istek başına geçici RAM, önbellek büyümesi, VRAM | `psutil` (VRAM için `nvidia-smi`) |
| `llm` | Model warmup, time-to-first-token, toplam üretim, token/s | **Çalışan Foundry Local + yüklü model** |
| `bundle` | Frontend chunk sayısı ve gzip boyutları | `frontend-web/node_modules` |
| `concurrent` | 1/2/5/10 eşzamanlı `/ask/stream`, p95, hata oranı, yük altında `/health` | **Çalışan backend** (`uvicorn`) |

Ortam eksikse suite **çökmez**, "atlandı" olarak işaretlenir. Kısmi bir
ortamda bile elde edilebilir her sayı elde edilir.

### `llm` ve `concurrent` için

```bash
# Foundry Local ayakta olmalı
foundry server start
foundry model load ministral-3-3b-instruct-2512

# concurrent için ayrıca backend gerekli (ayrı terminal)
uvicorn backend.app.main:app --port 8000

python -m benchmarks.bench_llm
python -m benchmarks.bench_concurrent --user-id 1
```

## Neden medyan ve p95 (ortalama değil)

Latency dağılımları çarpık: birkaç yavaş örnek ortalamayı yanıltıcı biçimde
yukarı çeker. Kullanıcının hissettiği şey medyan (tipik durum) ve p95
(kötü gün). Ham örnekler de JSON'a yazılıyor — sonradan farklı bir
istatistik gerekirse yeniden çalıştırmaya gerek kalmasın.

## Veri güvenliği

- Üretim veritabanı **hiçbir zaman** yazılmaz; ölçümler geçici bir kopya üzerinde çalışır.
- FAISS index'leri yalnızca **okunur**; yazma testleri geçici dizinde yapılır.
- `results/` klasörü versiyon kontrolüne girebilir (küçük JSON'lar) — önce/sonra
  karşılaştırmasının kaydı olarak değerlidir.

## Bilinen kısıtlar

- `token_per_second` kaba bir tahmin (~4 karakter/token). Modeller arası
  karşılaştırma için değil, **aynı model** üzerinde önce/sonra için anlamlı.
- `bench_concurrent` gerçek bir sunucuya istek atar (TestClient değil) —
  çünkü ölçmek istediğimiz şey tam olarak uvicorn'un event loop'u ve
  threadpool davranışı.
- GPU metrikleri `nvidia-smi` gerektirir; yoksa `null` geçilir.

## `results/sandbox-reference.json` nedir?

Faz 1'de, altyapının çalıştığını doğrulamak için alınmış bir **referans**
koşu — Beyza'nın makinesinde değil, bir Linux sandbox'ta alındı ve orada
embedding modeli ile Foundry Local **yoktu**. Yani:

- `retrieval`, `resources`, `bundle` sayıları geçerli ama **farklı donanımdan**;
- `llm`, `concurrent`, `ingestion` ve tüm embedding ölçümleri **atlandı**.

**Gerçek baseline bu değil.** Kendi makinende şunu çalıştır:

```bash
python -m benchmarks.run_all --label baseline
```

Bundan sonraki her karşılaştırma o dosyaya karşı yapılmalı.
