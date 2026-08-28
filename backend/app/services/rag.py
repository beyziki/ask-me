"""Hybrid RAG çekirdeği: Semantic Search (FAISS) + BM25.

Her kullanıcı için ayrı FAISS index dosyası tutulur (data/vectorstore/<user_id>.faiss).
BM25 tarafı kullanıcının chunk'ları üzerinden kurulur ve süreç içinde
önbelleklenir (bkz. `_get_bm25`); index dosyası da mtime'a bakılarak
önbellekte tutulur (bkz. `_read_index_cached`) — ikisi de eskiden HER
soruda sıfırdan yapılıyordu ve toplam gecikmenin gözle görülür bir kısmıydı.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path
from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.app.core.config import settings

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Embedding modelini (süreç başına bir kez) yükler.

    `device` AÇIKÇA veriliyor: verilmezse sentence-transformers CUDA varsa
    otomatik GPU'ya yerleşiyor ve Foundry Local'in dil modeliyle aynı VRAM'i
    paylaşmaya çalışıyor -- 8 GB'lık kartta bu, gözlemlenmiş sert bir
    "CudaMallocArray - out of memory" hatasına yol açıyor. Gerekçenin
    tamamı: `core/config.py:embedding_device`.
    """
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(
            settings.embedding_model, device=settings.embedding_device
        )
    return _embedder


# --- Tokenizasyon ---------------------------------------------------------
# Eskiden BM25 hem korpusu hem sorguyu ham `str.split()` ile ayırıyordu. Bu,
# arama kalitesindeki en büyük tek kayıptı:
#   * "Turing?" ile "Turing" ve "turing" ÜÇ FARKLI terim sayılıyordu, yani
#     soruda noktalama/büyük harf varsa eşleşme tamamen kaçıyordu;
#   * "makinesi," gibi virgüllü kelimeler hiçbir zaman eşleşmiyordu.
# Aşağıdaki tokenizasyon küçük harfe indirip yalnızca harf/rakam dizilerini
# alıyor.
#
# "i" AİLESİ ÖZEL DURUMU: Türkçe'de dört ayrı i harfi var (i, ı, İ, I) ve
# Python'un `lower()`'ı Türkçe kuralını bilmiyor ("I" -> "i"). Türkçe kuralı
# uygulamak (I -> ı) da işe yaramıyor, çünkü bu derste geçen İNGİLİZCE
# terimleri bozuyor: "TURING" -> "turıng", metindeki "Turing" -> "turing",
# yani eşleşme kaçıyor. Arama tarafında doğru çözüm, dört formu da TEK bir
# forma katlamak: "işlemci"/"İŞLEMCİ" ve "Turing"/"TURING" artık aynı
# terime iniyor. (Bu yalnızca ARAMA anahtarını etkiliyor; kullanıcıya
# gösterilen metin hiç değişmiyor.)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_I_FOLD_MAP = str.maketrans({"I": "i", "İ": "i", "ı": "i"})


def tokenize(text: str) -> list[str]:
    """BM25 için ortak tokenizasyon (hem korpus hem sorgu bunu kullanır)."""
    return _TOKEN_RE.findall(text.translate(_I_FOLD_MAP).lower())


def _index_path(user_id: int) -> Path:
    return settings.vectorstore_dir / f"user_{user_id}.faiss"


# --- FAISS index önbelleği ------------------------------------------------
# `semantic_search` her soruda `faiss.read_index` ile index'i DİSKTEN
# okuyordu. Index yalnızca dosya yükleme/silme sırasında değiştiği için,
# (yol, mtime, boyut) anahtarıyla önbelleğe alıp tekrar tekrar okumayı
# bırakıyoruz; dosya değişirse anahtar da değişeceği için önbellek kendini
# otomatik geçersiz kılıyor (bayat index okuma riski yok).
_index_cache: dict[str, tuple[tuple[float, int], faiss.Index]] = {}
_index_cache_lock = threading.Lock()


def _read_index_cached(path: Path) -> faiss.Index:
    stat = path.stat()
    stamp = (stat.st_mtime, stat.st_size)
    key = str(path)
    with _index_cache_lock:
        cached = _index_cache.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]
    index = faiss.read_index(key)
    with _index_cache_lock:
        _index_cache[key] = (stamp, index)
    return index


def _load_or_create_index(user_id: int, dim: int) -> faiss.Index:
    path = _index_path(user_id)
    if path.exists():
        return faiss.read_index(str(path))
    return faiss.IndexFlatIP(dim)  # cosine sim için normalize edilmiş vektörlerle IP kullanılır


def add_chunks_to_index(user_id: int, texts: list[str]) -> list[int]:
    """Yeni chunk'ları embedding'leyip FAISS index'ine ekler.

    Dönüş: her chunk'ın index içindeki satır numarası (vector_row).
    """
    if not texts:
        return []
    embedder = get_embedder()
    vectors = embedder.encode(texts, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype="float32")

    index = _load_or_create_index(user_id, vectors.shape[1])
    start_row = index.ntotal
    index.add(vectors)
    faiss.write_index(index, str(_index_path(user_id)))

    return list(range(start_row, start_row + len(texts)))


def rebuild_index(user_id: int, texts: list[str]) -> list[int]:
    """Kullanıcının FAISS index'ini SIFIRDAN yeniden oluşturur.

    Bir doküman silindiğinde (bkz. api/documents.py:delete_document)
    kullanılır. Düz `IndexFlatIP`, tek tek satır silmeyi desteklemiyor —
    bir satırı silip geri kalanları kaydırmak, o satırdan SONRAKİ tüm
    chunk'ların `vector_row`'unu (bkz. db/models.py:Chunk) geçersiz kılar.
    Bunun yerine en güvenli yol: silinen dokümana ait OLMAYAN tüm kalan
    chunk'ları (verilen sırayla) yeniden embed'leyip index'i baştan kurmak.
    Çağıran taraf, dönen `vector_row` listesini aynı sırayla kalan
    chunk'lara yazmalı.

    `texts` boşsa (kullanıcının hiç dokümanı kalmadıysa) eski index
    dosyasını siler — aksi halde yanlış boyutlu/bayat bir index dosyası
    kalıntı olarak kalır.
    """
    path = _index_path(user_id)
    if not texts:
        if path.exists():
            path.unlink()
        return []

    embedder = get_embedder()
    vectors = embedder.encode(texts, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(path))

    return list(range(len(texts)))


def drop_rows_from_index(user_id: int, keep_rows: list[int]) -> list[int] | None:
    """Index'i, YALNIZCA `keep_rows`'daki satırları koruyarak yeniden kurar.

    `rebuild_index`'in hızlı karşılığı. Aradaki fark kritik: `rebuild_index`
    kalan her chunk'ı SIFIRDAN yeniden embed'liyor -- oysa o vektörler zaten
    index dosyasının içinde duruyor. Bir dosya silindiğinde yüzlerce chunk'ı
    yeniden embed'lemek (CPU'da) dakikalar sürebiliyordu; gözlemlenen
    davranış "silme butonu bir türlü bitmiyor" şeklindeydi.

    Burada bunun yerine mevcut vektörleri `reconstruct_n` ile index'ten geri
    okuyup istenen satırları kopyalıyoruz — embedding modeline hiç
    dokunulmuyor, işlem saf bellek kopyasına iniyor.

    `keep_rows`: kalan chunk'ların ESKİ `vector_row` değerleri, korunmasını
    istediğin SIRAYLA. Dönüş: aynı sıradaki yeni `vector_row` değerleri
    (0..n-1). Hızlı yol uygulanamıyorsa (index dosyası yok, satır numaraları
    tutarsız) `None` döner — çağıran taraf `rebuild_index`'e düşmeli.
    """
    path = _index_path(user_id)
    if not path.exists():
        return None

    if not keep_rows:
        path.unlink()
        with _index_cache_lock:
            _index_cache.pop(str(path), None)
        return []

    index = _read_index_cached(path)
    # Satır numaraları index'le tutarsızsa (ör. index elle silinmiş, DB'deki
    # vector_row'lar bayat) sessizce yanlış vektör kopyalamak yerine hızlı
    # yoldan vazgeçip yeniden embed'lemeye düşüyoruz.
    if any(row is None or row < 0 or row >= index.ntotal for row in keep_rows):
        return None

    all_vectors = index.reconstruct_n(0, index.ntotal)
    vectors = np.asarray(all_vectors, dtype="float32")[keep_rows]

    new_index = faiss.IndexFlatIP(vectors.shape[1])
    new_index.add(vectors)
    faiss.write_index(new_index, str(path))

    return list(range(len(keep_rows)))


@dataclass
class SearchHit:
    chunk_id: int
    score: float


def _search_subset(
    index: faiss.Index, q_vec: np.ndarray, top_k: int, rows: np.ndarray
) -> list[tuple[int, float]]:
    """Yalnızca `rows` satirlari icinde arama yapar (dogru "filtreli" arama).

    Neden gerekli: `faiss.IndexFlatIP.search` her zaman TUM index uzerinde
    arar. Kullanici aramayi belirli dosyalarla sinirladiginda (bkz.
    `AskRequest.document_ids`) elimizde yalnizca o dosyalarin satirlari olur,
    ama index'e hala global bir arama yapiliyordu ve donen satirlardan
    kapsam disi olanlar SESSIZCE ATILIYORDU. Sonuc: seciligi dosyanin
    parcalari genel siralamada ilk `top_k` icine giremezse `semantic_search`
    BOS liste donuyordu -- yani "1 dosya secili" durumunda arama neredeyse
    her zaman bos kaliyordu. Bu, `api/ask.py:_retrieve`'de `best_semantic`
    degerini 0.0'a dusurup `has_context`'i False yapiyor ve modele hic
    context gonderilmiyordu; kullanici bunu "yuklenen dosyalarda bir bolum
    bulunamadi" mesaji ve dosyayla ilgisi olmayan bir cevap olarak
    goruyordu.

    Vektorler zaten index dosyasinin icinde oldugu icin, izin verilen
    satirlari `reconstruct_batch` ile geri okuyup dogrudan ic carpim
    aliyoruz -- vektorler normalize edildigi icin bu tam olarak kosinus
    benzerligi, yani global aramayla AYNI skor olcegi. Alt kume genelde
    kucuk oldugundan (bir-iki dosyanin parcalari) bu, tum index'i taramaktan
    daha da hizli.
    """
    try:
        vectors = index.reconstruct_batch(rows)
    except AttributeError:  # cok eski faiss surumleri icin geri dusus
        vectors = np.asarray(index.reconstruct_n(0, index.ntotal))[rows]
    vectors = np.asarray(vectors, dtype="float32")

    scores = vectors @ q_vec[0]
    order = np.argsort(-scores)[:top_k]
    return [(int(rows[i]), float(scores[i])) for i in order]


def semantic_search(user_id: int, query: str, top_k: int, row_to_chunk_id: dict[int, int]) -> list[SearchHit]:
    """`row_to_chunk_id`'de yer alan satirlar arasinda semantic arama.

    `row_to_chunk_id` cagiran tarafin arama KAPSAMIDIR: aramanin sinirlandigi
    dokumanlarin (hepsi seciliyse tum dokumanlarin) chunk'lari. Arama bu
    kapsamin ICINDE yapilir; kapsam disi satirlar hic puanlanmaz.
    """
    path = _index_path(user_id)
    if not path.exists() or not row_to_chunk_id:
        return []
    index = _read_index_cached(path)
    embedder = get_embedder()
    q_vec = embedder.encode([query], normalize_embeddings=True)
    q_vec = np.asarray(q_vec, dtype="float32")

    # DB'deki `vector_row` degerleri index ile tutarsiz olabilir (ornegin
    # index elle silinip yeniden kurulmussa); kapsam disi satirlari burada
    # eliyoruz ki `reconstruct_batch` patlamasin.
    allowed = np.array(
        sorted(row for row in row_to_chunk_id if 0 <= row < index.ntotal),
        dtype="int64",
    )
    if allowed.size == 0:
        return []

    if allowed.size == index.ntotal:
        # Kapsam tum index: FAISS'in kendi (optimize) aramasi.
        scores, rows = index.search(q_vec, min(top_k, index.ntotal))
        scored = [
            (int(row), float(score))
            for score, row in zip(scores[0], rows[0])
            if row != -1
        ]
    else:
        scored = _search_subset(index, q_vec, top_k, allowed)

    hits = []
    for row, score in scored:
        chunk_id = row_to_chunk_id.get(row)
        if chunk_id is not None:
            hits.append(SearchHit(chunk_id=chunk_id, score=score))
    return hits


# --- BM25 önbelleği -------------------------------------------------------
# Eskiden her soruda kullanıcının TÜM chunk'ları yeniden tokenize edilip
# `BM25Okapi(corpus)` sıfırdan kuruluyordu. Bu, doküman sayısıyla doğrusal
# büyüyen ve her soruda tekrar ödenen bir maliyet. Korpus yalnızca dosya
# yükleme/silme ile değiştiği için, chunk kimliklerinden türetilen bir
# imzayla önbelleğe alıyoruz: imza değişirse önbellek kendiliğinden
# geçersizleşiyor.
#
# VARSAYIM: `chunk_id` veritabanında birincil anahtar ve bir chunk'ın metni
# oluşturulduktan sonra DEĞİŞMİYOR (dosya güncellenmek istenirse silinip
# yeniden yükleniyor). Bu yüzden kimlik listesi, korpus içeriği için
# yeterli bir imza. Metinleri hash'lemek daha "güvenli" olurdu ama tüm
# korpusu her istekte yeniden okumak demekti -- yani kaçınmaya çalıştığımız
# maliyetin ta kendisi.
_bm25_cache: dict[int, tuple[list[int], BM25Okapi]] = {}
_bm25_cache_lock = threading.Lock()
_BM25_CACHE_MAX_ENTRIES = 32


def _corpus_signature(chunk_ids: list[int]) -> int:
    return hash(tuple(chunk_ids))


def _get_bm25(chunk_ids: list[int], corpus_texts: list[str]) -> BM25Okapi:
    signature = _corpus_signature(chunk_ids)
    with _bm25_cache_lock:
        cached = _bm25_cache.get(signature)
        if cached is not None and cached[0] == chunk_ids:
            return cached[1]
    bm25 = BM25Okapi([tokenize(text) for text in corpus_texts])
    with _bm25_cache_lock:
        if len(_bm25_cache) >= _BM25_CACHE_MAX_ENTRIES:
            _bm25_cache.clear()  # basit ve yeterli: küçük, sınırlı bir önbellek
        _bm25_cache[signature] = (chunk_ids, bm25)
    return bm25


def bm25_search(chunk_id_to_text: dict[int, str], query: str, top_k: int) -> list[SearchHit]:
    if not chunk_id_to_text:
        return []
    chunk_ids = list(chunk_id_to_text.keys())
    corpus_tokens = [tokenize(chunk_id_to_text[cid]) for cid in chunk_ids]
    bm25 = _get_bm25(chunk_ids, [chunk_id_to_text[cid] for cid in chunk_ids])
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    scores = bm25.get_scores(query_tokens)

    # Sorgu terimlerinden HİÇBİRİ geçmeyen parçaları eliyoruz. Eskiden
    # böyle parçalar da listeye giriyor, ardından normalizasyon en yüksek
    # skoru -- alakasız olsa bile -- 1.0'a çekip `hybrid_merge`'te gerçekten
    # ilgili semantic sonuçların önüne geçiriyordu. Bu, kullanıcının
    # şikâyet ettiği "konuyla ilgisiz kaynaklardan cevap" davranışının
    # doğrudan sebeplerinden biriydi.
    # (Doğrudan `score > 0` filtresi kullanılmıyor: BM25Okapi'nin IDF'i çok
    # küçük korpuslarda geçerli eşleşmeler için bile 0 dönebiliyor.)
    query_token_set = set(query_tokens)
    candidates = [
        (cid, score)
        for cid, score, tokens in zip(chunk_ids, scores, corpus_tokens)
        if query_token_set.intersection(tokens)
    ]
    if not candidates:
        return []

    ranked = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
    max_score = max(score for _, score in ranked) or 1.0
    return [SearchHit(chunk_id=cid, score=score / max_score) for cid, score in ranked]


def rrf_merge(
    semantic_hits: list[SearchHit],
    bm25_hits: list[SearchHit],
    k: int | None = None,
) -> list[SearchHit]:
    """Semantic ve BM25 sonuçlarını Reciprocal Rank Fusion ile birleştirir.

    NEDEN `hybrid_merge` YERİNE BU: eski ağırlıklı skor toplamı, iki listenin
    skorlarının KARŞILAŞTIRILABİLİR olduğunu varsayıyordu — ama değiller.
    Semantic taraf gerçek bir kosinüs benzerliği (tipik olarak 0.3-0.8),
    BM25 tarafı ise kendi içinde en yükseğe göre normalize edilmiş bir
    değerdi; yani BM25'in en iyi adayı, ALAKASIZ olsa bile her zaman 1.0
    alıyordu. alpha=0.5 ile bu, alakasız bir BM25 sonucuna 0.5 puan verip
    gerçekten ilgili bir semantic sonucu (ör. 0.45) geçmesine yol açıyordu —
    kullanıcının gördüğü "konuyla ilgisiz kaynaklardan cevap üretme"
    davranışının doğrudan sebeplerinden biri.

    RRF skor değerlerine değil yalnızca SIRAYA bakar (1/(k+rank)), bu yüzden
    iki farklı ölçeği güvenle birleştirir ve her iki listede de üst sıralarda
    çıkan parçaları doğal olarak öne alır. `k` (varsayılan 60) tek bir
    listedeki uç sıralamaların etkisini yumuşatır.
    """
    if k is None:
        k = settings.rrf_k

    combined: dict[int, float] = {}
    for hits in (semantic_hits, bm25_hits):
        for rank, hit in enumerate(hits, start=1):
            combined[hit.chunk_id] = combined.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [SearchHit(chunk_id=cid, score=score) for cid, score in ranked]


def hybrid_merge(
    semantic_hits: list[SearchHit],
    bm25_hits: list[SearchHit],
    alpha: float = settings.hybrid_alpha,
) -> list[SearchHit]:
    """Semantic ve BM25 skorlarını alpha ağırlığıyla birleştirir (0=BM25, 1=semantic).

    ESKİ birleştirme yöntemi; `/ask` artık `rrf_merge` kullanıyor (nedeni
    orada anlatıldı). Burada duruyor çünkü hâlâ testleri var ve alpha ile
    ağırlıklandırma davranışı karşılaştırma/deney için işe yarıyor.
    """
    combined: dict[int, float] = {}
    for hit in semantic_hits:
        combined[hit.chunk_id] = combined.get(hit.chunk_id, 0.0) + alpha * hit.score
    for hit in bm25_hits:
        combined[hit.chunk_id] = combined.get(hit.chunk_id, 0.0) + (1 - alpha) * hit.score

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [SearchHit(chunk_id=cid, score=score) for cid, score in ranked]
