# RAG Retrieval Eval

**"Cevaplar daha iyi oldu mu?" sorusunun tek objektif cevabı burası.**

Hız benchmark'ları (`benchmarks/`) bir değişikliğin hızlandırdığını gösterir
ama **doğruluğu bozup bozmadığını göstermez**. Bir RAG sisteminde bu en
tehlikeli hata: context'i kısaltarak her zaman hızlanabilirsin, karşılığında
cevap kalitesini kaybedersin. Bu klasör o değiş-tokuşu görünür kılıyor.

## Çalıştırma

```bash
python -m evals.run_eval --label baseline

# ... bir değişiklik yap ...

python -m evals.run_eval --label faz2
python -m evals.compare baseline faz2
```

`compare`, herhangi bir soru **bozulursa** çıkış kodu 1 döner.

LLM'e hiç gitmez — saniyeler sürer, deterministiktir, her değişiklikte
çalıştırılabilir.

## Neden LLM yok

Ölçülen şey "model iyi yazdı mı" değil, **"doğru kaynakları buldu mu"**.
İkisini karıştırmak iki sorun yaratır: model örneklemesi deterministik
değildir (aynı değişiklik iki kez farklı sonuç verir) ve her değerlendirme
dakikalar sürer (kimse çalıştırmaz).

## İki katman, ve aradaki boşluk

| Katman | Ne | Nereden |
|---|---|---|
| **Ham sıralama** | Arama doğru dokümanı bulabildi mi? | `semantic_search` + `bm25_search` + `rrf_merge` |
| **Teslim edilen** | Kullanıcıya gerçekten ulaştı mı? | `_retrieve` — ham sıralama + `min_relevance_score` + dedupe + `_cap_context` |

**Aradaki fark en önemli sayı.** Ham sıralama doğru dokümanı bulup teslim
edilen sonuç bulmuyorsa sorun aramada değil, eşikte veya kırpmadadır — ve
tamamen farklı bir düzeltme gerektirir.

## Metrikler

- `recall@k` — beklenen dokümanlardan en az biri sonuçlarda mı
- `precision@k` — sonuçların kaçı beklenen dokümandan
- `MRR` — ilk doğru dokümanın sırasının tersi (üst sıraya çıkarmayı ödüllendirir)
- `term_hit` — ayırt edici terimlerden biri getirilen metinde geçiyor mu
- `has_context_accuracy` — bayrak doğru mu; ayrıca:
  - `false_no_context_rate` — içerik VARKEN "bulunamadı" denmesi (kullanıcının gördüğü sahte uyarı)
  - `false_has_context_rate` — içerik YOKKEN "bulundu" denmesi (uydurma riski)
- `semantic_hit_count` — **audit P0-1'in doğrudan göstergesi.** Doküman filtresi
  seçildiğinde bu sayı düşüyorsa, filtre semantic aramayı fiilen kapatıyor demektir.

## Ground truth neden chunk id değil

Doğru cevap "hangi **chunk**" olarak değil, "hangi **doküman** + hangi
**terimler**" olarak yazıldı.

Sebep pratik: Faz 2'de `chunk_size` değişip tüm korpus yeniden indekslenecek
(audit P0-5) ve o an **bütün chunk id'leri değişecek**. Chunk id'ye bağlı bir
ground truth o değişiklikten sonra kullanılamaz hâle gelirdi — yani tam da
ölçmek istediğimiz iyileştirmeyi ölçemezdik. Doküman + terim seviyesi yeniden
chunking'den etkilenmiyor.

## Vaka grupları

| Önek | Ne ölçüyor |
|---|---|
| (yok) | Normal sorular, tüm dosyalarda arama |
| `f-`, `f2-` | **Doküman filtreli** — audit P0-1'i doğrudan ölçüyor. `f-net-only` kontrol grubu (büyük doküman, zarar görmemeli) |
| `nc-` | Korpusla ilgisiz sorular — `has_context=false` beklenir |

## Bu set bir TASLAK

Sorular `data/ask_me.db`'deki gerçek doküman içeriğinden türetildi ve
`expected_documents` alanları içerik taramasıyla doğrulandı — ama **insan
doğrulaması yapılmadı.**

Lütfen `dataset.json`'daki vakaları gözden geçir:
- Soru anlamlı mı?
- `expected_documents` doğru mu?
- `required_terms` gerçekten o dokümana özgü mü?
- Eklemek istediğin sorular var mı?

Yanlış bir ground truth, ondan sonraki her ölçümü sessizce bozar.

## Bilinen kısıt

`Chapter_6.pptx` (doküman 2) korpusun %29'unu kaplıyor ve içeriği tamamen
bozuk (bkz. audit — `.pptx` binary olarak indekslenmiş). Bu dosya hakkında
soru YOK; ama owner 1'in tüm sorularında bir gürültü kaynağı olarak
duruyor — yani eval, bu sorunun düzeltilmesinin etkisini de ölçecek.
