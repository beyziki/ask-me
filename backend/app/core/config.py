"""Uygulama genel ayarları."""
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parents[3]

# Akıl yürütme ("thinking") modu olan model aileleri. Bu modeller cevaptan
# önce bir `<think>...</think>` bloğu üretir; Foundry Local bunu kapatmanın
# resmî bir yolunu sunmuyor (bkz. services/llm.py'deki not), bu yüzden
# yalnızca token bütçesini geniş tutup bloğu çıktıdan temizleyebiliyoruz.
# Düşünmeyen (instruct) modellerde bu maliyet tamamen ortadan kalkıyor:
# hem çok daha hızlı, hem de "boş cevap" arızası hiç oluşmuyor.
_THINKING_MODEL_MARKERS = ("qwen3", "deepseek-r1", "qwq", "-thinking")


class Settings(BaseSettings):
    app_name: str = "Ask Me? - Offline AI Study Assistant"
    data_dir: Path = BASE_DIR / "data"
    uploads_dir: Path = BASE_DIR / "data" / "uploads"
    vectorstore_dir: Path = BASE_DIR / "data" / "vectorstore"
    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'ask_me.db'}"

    # --- Foundry Local -----------------------------------------------------
    # `.env` içinde FOUNDRY_MODEL_ALIAS ile değiştirilebilir. Varsayılan
    # olarak DÜŞÜNMEYEN (instruct) bir model kullanıyoruz: Qwen3-8B'nin
    # thinking modu Foundry Local üzerinden kapatılamadığı için her soruda
    # yüzlerce-binlerce token'ı görünmeyen akıl yürütmeye harcıyordu — hem
    # yavaşlığın hem de gözlemlenen "boş cevap"/"tek nokta" arızasının ana
    # kaynağı buydu.
    # Alias'ların katalogdaki tam adı `foundry model list` ile doğrulanmalı;
    # burada yazan isim o listedekiyle birebir aynı olmalı.
    #
    # MODEL BOYUTU / VRAM: 12B'lik bir model (6.6 GB) 8 GB VRAM'li bir kartta
    # yükleniyor ama KV cache ile birlikte kartı taşırıp CUDA bağlamını
    # bozuyor ("illegal memory access", bkz. services/llm.py:GpuContextLost).
    # Bu bağlam kaybı süreç içinde kurtarılamıyor. Bu yüzden varsayılan,
    # rahat sığan bir modele çekildi. Güçlü bir kartın varsa .env'den daha
    # büyüğüne geçebilirsin.
    foundry_model_alias: str = "ministral-3-3b-instruct-2512"
    # None -> alias'tan otomatik çıkarılır (bkz. `model_has_thinking`).
    # Elle zorlamak için .env'de MODEL_THINKING=true/false yazılabilir.
    model_thinking: bool | None = None

    # Cevap üretimi için token bütçesi. Düşünmeyen modelde tek ve küçük bir
    # bütçe yeterli (üretilen her token görünür cevap). Thinking modelde ilk
    # deneme başarısız olursa çok daha geniş bir bütçeyle tekrar denenir.
    answer_max_tokens: int = 800
    answer_max_tokens_retry: int = 4096

    # Modeli backend açılırken arka planda ısıt: ilk sorunun soğuk başlangıç
    # gecikmesini kullanıcıdan gizler. Testlerde/CI'da kapatılabilir.
    warmup_on_startup: bool = True

    # --- Özet (bkz. services/summary.py) -----------------------------------
    # Özet, dokümanın TAMAMINI modelden geçirir. Uzun dosyalarda bu tek
    # istekte yapılamayacağı için doküman gruplara bölünüp önce her grup
    # ayrı özetlenir (map), sonra ara özetler birleştirilir (reduce).
    # Bir map grubunun karakter üst sınırı. Bu değer İYİMSER bir başlangıç:
    # model kapasiteye çarparsa (bkz. services/summary.py:_is_capacity_error —
    # 12B model + 8GB VRAM'de gözlemlenen "CUDA illegal memory access")
    # ilgili grup otomatik olarak ikiye bölünüp tekrar deneniyor. Yani bu
    # sayıyı düşürmek gerekmiyor; düşürmek yalnızca ilk denemede çarpma
    # olasılığını azaltır, karşılığında tur sayısını artırır.
    summary_group_max_chars: int = 4000
    # Nihai (kullanıcıya gösterilen) özetin token bütçesi.
    summary_max_tokens: int = 1200
    # Ara özetlerin bütçesi: bunlar kullanıcıya gösterilmiyor, yalnızca
    # birleştirme adımına girdi — kısa ve yoğun olmaları yeterli.
    summary_partial_max_tokens: int = 500

    # --- RAG ---------------------------------------------------------------
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Embedding modelinin çalışacağı cihaz. VARSAYILAN BİLİNÇLİ OLARAK "cpu".
    #
    # NEDEN: sentence-transformers cihaz verilmediğinde CUDA varsa otomatik
    # GPU'ya yerleşiyor. Ama GPU'yu zaten Foundry Local'in dil modeli
    # kullanıyor (3.6 GB + KV cache) ve bu kart 8 GB. İkisi aynı anda
    # yerleşmeye çalıştığında ONNX Runtime GenAI tarafında sert bir hata
    # alınıyor -- gözlemlendi:
    #
    #   CUDA error in CudaMallocArray ... - out of memory
    #
    # Embedding modeli küçük (MiniLM, 384 boyut) ve soru başına TEK bir kısa
    # metin gömüyor; bu iş CPU'da milisaniyeler sürüyor. Karşılığında
    # LLM'e ayrılan VRAM'i boşaltıyor -- yani gecikmenin asıl olduğu yere.
    #
    # TAKAS: toplu gömme (dosya yükleme sırasında yüzlerce chunk) CPU'da
    # daha yavaş. Yükleme süresi seni rahatsız ederse ölç
    # (`python -m benchmarks.bench_ingestion`) ve kartında yer varsa
    # `.env`'de EMBEDDING_DEVICE=cuda yap.
    embedding_device: str = "cpu"
    # Chunk boyutu KELİME cinsinden. 500 kelimelik (~700+ token) parçalar hem
    # alma (retrieval) hassasiyetini düşürüyordu (tek parça birden çok konuyu
    # kapsıyor, embedding'i bulanıklaşıyor) hem de 10 parçalık bir context'i
    # ~7000 token'a çıkarıp üretimi ciddi biçimde yavaşlatıyordu.
    chunk_size: int = 220
    chunk_overlap: int = 40
    top_k_semantic: int = 8
    top_k_bm25: int = 8
    # Birleştirme sonrası LLM'e gerçekten gönderilen parça sayısı.
    max_context_chunks: int = 5
    # Context'in toplam karakter üst sınırı (güvenlik freni: çok uzun
    # parçalar prefill süresini uçuruyor).
    max_context_chars: int = 6000
    hybrid_alpha: float = 0.5  # yalnızca eski `hybrid_merge` için (bkz. rag.py)
    # Reciprocal Rank Fusion sabiti (literatürdeki standart değer 60).
    rrf_k: int = 60
    # En iyi semantic benzerlik bu eşiğin altındaysa "kaynaklarda ilgili
    # bilgi yok" kabul edilir; model alakasız parçalardan cevap uydurmak
    # yerine genel bilgiyle ve açık bir uyarıyla cevap verir (bkz. api/ask.py).
    min_relevance_score: float = 0.30

    # --- Dil ---------------------------------------------------------------
    supported_languages: tuple[str, ...] = ("tr", "en")

    class Config:
        env_file = ".env"

    @property
    def model_has_thinking(self) -> bool:
        """Seçili modelin akıl yürütme (thinking) modu var mı?

        `.env`'de MODEL_THINKING açıkça verilmişse o kullanılır; verilmemişse
        alias'tan çıkarılır. Bu bayrak `services/llm.py`'de üç şeyi belirler:
        prompt'a `/no_think` eklenip eklenmeyeceğini, `enable_thinking=False`
        denemesinin yapılıp yapılmayacağını ve token bütçesi stratejisini.
        """
        if self.model_thinking is not None:
            return self.model_thinking
        alias = self.foundry_model_alias.lower()
        return any(marker in alias for marker in _THINKING_MODEL_MARKERS)

    @property
    def answer_token_budgets(self) -> list[int]:
        """Cevap üretiminde sırayla denenecek `max_tokens` bütçeleri.

        Düşünmeyen modelde tek bütçe yeterli: model görünür cevaba hemen
        başladığı için "tüm bütçeyi akıl yürütmede tüketme" arızası yok.
        Thinking modelde ise ilk (ucuz) bütçe tükenirse çok daha geniş bir
        bütçeyle GERÇEK bir eskalasyon yapılır — aynı bütçeyle tekrar
        denemenin faydası olmadığı gözlemlendi.
        """
        if not self.model_has_thinking:
            return [self.answer_max_tokens]
        return [max(self.answer_max_tokens, 1500), self.answer_max_tokens_retry]


settings = Settings()
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
