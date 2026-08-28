"""Ask Me? - FastAPI giriş noktası."""
import logging
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.db.base import Base, engine, run_startup_migrations
from backend.app.db import models  # noqa: F401 - tabloların oluşması için import edilir
from backend.app.api import ask, code, documents, quiz, summary, users

# Uygulama loglarini GORUNUR yap.
#
# Uvicorn yalnizca KENDI logger'larini yapilandiriyor; uygulama modullerinin
# `logging.getLogger(__name__)` ile aldigi logger'lar kok logger'a dusuyor ve
# kok logger'in hicbir handler'i olmadigi icin Python'un "lastResort"
# mekanizmasi devreye giriyor -- o da yalnizca WARNING ve ustunu basiyor.
# Sonuc: `logger.info(...)` ile yazilan her sey SESSIZCE KAYBOLUYORDU, ki
# teshis icin eklenen retrieval satiri (bkz. api/ask.py:_retrieve) tam olarak
# o seviyedeydi ve kimse goremiyordu.
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(levelname)s:     %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)
# create_all yeni tabloları oluşturur ama var olan tablolara sonradan
# eklenen sütunları eklemez (bkz. db/base.py:run_startup_migrations).
run_startup_migrations()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # yerel geliştirme için; production'da daraltılmalı
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router)
app.include_router(documents.router)
app.include_router(ask.router)
app.include_router(quiz.router)
app.include_router(summary.router)
app.include_router(code.router)


def _warmup() -> None:
    """Embedding modelini ve Foundry Local'i arka planda hazırlar.

    Bu iki şey de eskiden İLK kullanıcı isteğinin içinde, tembel (lazy)
    olarak yükleniyordu:
      * `sentence-transformers` modeli ilk soruda/ilk yüklemede diskten
        okunup belleğe alınıyordu (birkaç saniye),
      * `foundry server start` + `foundry model load` ilk soruda çalışıyordu
        (soğuk başlangıçta onlarca saniye).
    Sonuç, ilk sorunun çok uzun sürmesi ve uygulamanın "yavaş" görünmesiydi.
    Burada backend açılır açılmaz, ayrı bir thread'de (uvicorn'un başlamasını
    geciktirmeden) ikisini de ısıtıyoruz.
    """
    try:
        from backend.app.services.rag import get_embedder

        get_embedder()
        logger.info("Embedding modeli hazır.")
    except Exception as exc:  # pragma: no cover - ortama bağlı
        logger.warning("Embedding modeli ön ısıtılamadı: %s", exc)

    try:
        from backend.app.services.llm import warmup as llm_warmup

        llm_warmup()
    except Exception as exc:  # pragma: no cover - ortama bağlı
        logger.warning("LLM ön ısıtma başarısız: %s", exc)


@app.on_event("startup")
def on_startup() -> None:
    # Testlerde/CI'da WARMUP_ON_STARTUP=false ile kapatılabilir (bkz. config.py).
    if not settings.warmup_on_startup:
        return
    threading.Thread(target=_warmup, name="ask-me-warmup", daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}


@app.get("/health/model")
def model_health():
    """Seçili modelin ve thinking modunun ne olduğunu gösterir.

    Hangi modelin gerçekten kullanıldığını (ve `/no_think` mantığının devrede
    olup olmadığını) arayüzden/terminalden hızlıca doğrulamak için.
    """
    return {
        "model_alias": settings.foundry_model_alias,
        "thinking_mode": settings.model_has_thinking,
        "answer_token_budgets": settings.answer_token_budgets,
    }
