"""Hybrid RAG'in skor birleştirme (hybrid_merge) ve BM25 arama mantığı için
birim testleri. Bunlar embedding modeli/FAISS'e ihtiyaç duymuyor, bu yüzden
ağ erişimi olmayan ortamlarda da (ör. CI) hızlıca çalışabiliyor.
"""
from backend.app.services.rag import (
    SearchHit,
    bm25_search,
    hybrid_merge,
    rrf_merge,
    tokenize,
)


def test_hybrid_merge_combines_scores_with_alpha():
    semantic = [SearchHit(chunk_id=1, score=0.9), SearchHit(chunk_id=2, score=0.4)]
    bm25 = [SearchHit(chunk_id=2, score=1.0), SearchHit(chunk_id=3, score=0.5)]
    merged = hybrid_merge(semantic, bm25, alpha=0.5)
    ids = [h.chunk_id for h in merged]
    # chunk 2 hem semantic hem bm25'te göründüğü için en yüksek birleşik
    # skoru almalı (0.5*0.4 + 0.5*1.0 = 0.7 > diğerleri)
    assert ids[0] == 2


def test_hybrid_merge_alpha_zero_only_uses_bm25():
    semantic = [SearchHit(chunk_id=1, score=1.0)]
    bm25 = [SearchHit(chunk_id=2, score=1.0)]
    merged = hybrid_merge(semantic, bm25, alpha=0.0)
    scores = {h.chunk_id: h.score for h in merged}
    assert scores[1] == 0.0
    assert scores[2] == 1.0


def test_bm25_search_ranks_relevant_chunk_higher():
    chunks = {
        1: "Turing makinesi hesaplanabilirlik teorisinin temelidir.",
        2: "Bu paragrafın konusuyla hiçbir ilgisi yok, tamamen alakasız bir metin.",
    }
    hits = bm25_search(chunks, "Turing makinesi nedir", top_k=2)
    assert hits[0].chunk_id == 1


def test_bm25_search_handles_empty_corpus():
    assert bm25_search({}, "sorgu", top_k=5) == []


# --- Tokenizasyon ---------------------------------------------------------


def test_tokenize_lowercases_and_strips_punctuation():
    # Eskiden ham `str.split()` kullanılıyordu: "Turing?" ile "turing" farklı
    # terim sayılıyor, noktalama içeren sorularda BM25 eşleşmesi tamamen
    # kaçıyordu.
    assert tokenize("Turing makinesi, nedir?") == ["turing", "makinesi", "nedir"]


def test_tokenize_folds_all_four_turkish_i_forms_together():
    # i / ı / İ / I dört ayrı harf; arama anahtarında hepsi tek forma
    # inmeli ki hem "İŞLEMCİ" ile "işlemci" hem de "TURING" ile "Turing"
    # eşleşsin (bkz. rag.py'deki not).
    assert tokenize("İşlemci") == tokenize("işlemci") == tokenize("İŞLEMCİ")
    assert tokenize("TURING") == tokenize("Turing") == ["turing"]
    assert tokenize("ışık") == tokenize("IŞIK")


def test_bm25_search_matches_despite_punctuation_and_case():
    # NOT: chunk id'leri testler arasında bilerek farklı tutuluyor -- BM25
    # önbelleği (bkz. rag.py:_get_bm25) korpusu chunk id listesiyle
    # anahtarlıyor ve id'ler gerçek veritabanında birincil anahtar.
    chunks = {101: "Turing makinesi hesaplanabilirlik teorisinin temelidir."}
    assert bm25_search(chunks, "TURING?", top_k=2)[0].chunk_id == 101


def test_bm25_search_returns_nothing_when_no_query_term_matches():
    # Sorgu terimlerinden hiçbiri geçmiyorsa BM25 aday döndürmemeli:
    # eskiden döndürülüyor ve normalizasyon bu alakasız adayı 1.0'a çekip
    # gerçekten ilgili semantic sonuçların önüne geçiriyordu.
    chunks = {102: "Derleyici tasarımı ve sözdizimi ağaçları."}
    assert bm25_search(chunks, "fotosentez klorofil", top_k=5) == []


# --- Reciprocal Rank Fusion ----------------------------------------------


def test_rrf_merge_ranks_items_found_in_both_lists_first():
    semantic = [SearchHit(chunk_id=1, score=0.9), SearchHit(chunk_id=2, score=0.4)]
    bm25 = [SearchHit(chunk_id=2, score=1.0), SearchHit(chunk_id=3, score=0.5)]
    merged = rrf_merge(semantic, bm25)
    # chunk 2 her iki listede de var -> iki katkı toplanır, başa gelmeli.
    assert merged[0].chunk_id == 2
    assert {h.chunk_id for h in merged} == {1, 2, 3}


def test_rrf_merge_ignores_incomparable_score_scales():
    # RRF'in asıl faydası: BM25'in normalize edilmiş 1.0'ı, semantic'in
    # gerçek kosinüs skorunu EZMEMELİ. Burada semantic'in en iyi adayı (10)
    # BM25'in en iyi adayından (20) daha yüksek sırada olduğu için ilk
    # sırayı almalı -- skor değerleri ne olursa olsun.
    semantic = [SearchHit(chunk_id=10, score=0.42)]
    bm25 = [SearchHit(chunk_id=20, score=1.0)]
    merged = rrf_merge(semantic, bm25)
    assert merged[0].chunk_id == 10  # eşit sırada, ilk liste öncelikli
    assert len(merged) == 2


def test_rrf_merge_handles_empty_inputs():
    assert rrf_merge([], []) == []
    only_semantic = rrf_merge([SearchHit(chunk_id=1, score=0.5)], [])
    assert [h.chunk_id for h in only_semantic] == [1]
