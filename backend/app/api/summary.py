"""Doküman özeti uç noktaları.

Özet, dokümanın tamamından üretilir (bkz. services/summary.py) ve doküman
başına EN FAZLA bir tane olacak şekilde veritabanında saklanır. Kalıcı
olmasının iki nedeni var: (1) üretimi pahalı, tekrar açıldığında beklemek
gereksiz; (2) "özetten quiz üret" akışı (bkz. api/quiz.py) hazır metni
kullanarak ham parçalardan üretmeye göre çok daha hızlı çalışıyor.
"""
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, get_current_user_id
from backend.app.core.config import settings
from backend.app.db.base import SessionLocal
from backend.app.db.models import Chunk, Document, Summary, User
from backend.app.models.schemas import SummaryOut, SummaryStatus
from backend.app.services.llm import (
    DegenerateOutput,
    EmptyAnswer,
    FoundryNotAvailable,
    GpuContextLost,
)
from backend.app.services.summary import generate_summary_stream

router = APIRouter(prefix="/summary", tags=["summary"])
logger = logging.getLogger(__name__)


def _owned_document(document_id: int, user_id: int, db: Session) -> Document:
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.owner_id == user_id)
        .first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Doküman bulunamadı")
    return document


@router.get("", response_model=list[SummaryStatus])
def list_summary_status(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Kullanıcının hangi dokümanlarının özeti olduğunu döner.

    Arayüz bunu, dosya listesinde "özeti var" işareti göstermek ve Quiz
    ekranında "özetten üret" seçeneğini aktif edip etmemek için kullanıyor —
    her dosya için ayrı ayrı GET atmaya gerek kalmıyor.
    """
    rows = (
        db.query(Summary.document_id)
        .filter(Summary.owner_id == user_id)
        .all()
    )
    with_summary = {row[0] for row in rows}
    documents = db.query(Document.id).filter(Document.owner_id == user_id).all()
    return [
        SummaryStatus(document_id=doc_id, has_summary=doc_id in with_summary)
        for (doc_id,) in documents
    ]


@router.get("/{document_id}", response_model=SummaryOut)
def get_summary(
    document_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    document = _owned_document(document_id, user_id, db)
    summary = db.query(Summary).filter(Summary.document_id == document_id).first()
    if not summary:
        raise HTTPException(status_code=404, detail="Bu doküman için henüz özet üretilmedi")
    return SummaryOut(
        document_id=document_id,
        filename=document.filename,
        content=summary.content,
        model_alias=summary.model_alias,
        updated_at=summary.updated_at,
    )


@router.delete("/{document_id}", status_code=204)
def delete_summary(
    document_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Özeti siler. Kullanıcı özeti beğenmezse ya da modeli değiştirdiyse
    sıfırdan ürettirebilsin diye."""
    _owned_document(document_id, user_id, db)
    summary = db.query(Summary).filter(Summary.document_id == document_id).first()
    if summary:
        db.delete(summary)
        db.commit()


def _load_document_chunks(document_id: int, user_id: int, db: Session) -> tuple[str, list[str], str]:
    """Dokümanın TÜM parçalarını sırayla yükler.

    `/quiz`'ten farklı olarak burada örnekleme YAPILMIYOR: özet dokümanın
    tamamını kapsamak zorunda. Uzunluk sorunu, parçaları gruplayıp map-reduce
    ile çözülüyor (bkz. services/summary.py:group_chunks).
    """
    document = _owned_document(document_id, user_id, db)

    chunks = (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
        .all()
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="Bu dokümanda işlenmiş içerik yok")

    user = db.query(User).get(user_id)
    filename = document.filename
    language = user.preferred_language
    texts = [c.content for c in chunks]

    # LLM çağrısı dakikalarca sürebiliyor; bu süre boyunca havuzdan bir
    # bağlantıyı (ve SQLite okuma kilidini) tutmamak için session'ı burada
    # kapatıyoruz — ask.py ve quiz.py'deki aynı düzeltme.
    db.close()

    return filename, texts, language


def _persist_summary(user_id: int, document_id: int, content: str) -> None:
    """Özeti kaydeder; zaten varsa ÜZERİNE YAZAR (doküman başına tek özet).

    Uzun süren LLM çağrısından sonra çalıştığı için kısa ömürlü, ayrı bir
    session kullanıyor (bkz. quiz.py:_persist_quiz'teki aynı not).
    """
    fresh_db = SessionLocal()
    try:
        summary = fresh_db.query(Summary).filter(Summary.document_id == document_id).first()
        if summary:
            summary.content = content
            summary.model_alias = settings.foundry_model_alias
        else:
            fresh_db.add(
                Summary(
                    owner_id=user_id,
                    document_id=document_id,
                    content=content,
                    model_alias=settings.foundry_model_alias,
                )
            )
        fresh_db.commit()
    finally:
        fresh_db.close()


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/{document_id}/stream")
def create_summary_stream(
    document_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Özeti canlı (SSE) üretir ve bitince veritabanına kaydeder.

    Olay sırası: (birden çok) `progress` ve `token` -> `done`; hata olursa
    `error`.

    `progress` olayları yalnızca UZUN dokümanlarda gelir: doküman tek bir
    gruba sığmıyorsa önce her bölüm ayrı özetlenir (map adımı) ve bu sırada
    kullanıcıya gösterilecek bir metin YOKTUR — ilerleme olayları o sessiz
    süreyi anlaşılır kılıyor. Kısa dokümanlarda map adımı hiç çalışmaz ve
    nihai özet doğrudan token token akar.
    """
    filename, texts, language = _load_document_chunks(document_id, user_id, db)

    def event_stream():
        collected: list[str] = []
        try:
            for kind, payload in generate_summary_stream(texts, language):
                if kind == "progress":
                    yield _sse({"type": "progress", "detail": payload})
                else:
                    collected.append(payload)
                    yield _sse({"type": "token", "content": payload})
        except (FoundryNotAvailable, DegenerateOutput, EmptyAnswer, GpuContextLost) as exc:
            logger.warning("Özet üretilemedi (%s): %s", type(exc).__name__, exc)
            yield _sse({"type": "error", "detail": str(exc)})
            return
        except Exception:
            logger.exception("/summary/%s/stream sırasında beklenmeyen hata", document_id)
            yield _sse(
                {"type": "error", "detail": "Özet üretilirken beklenmeyen bir hata oluştu."}
            )
            return

        content = "".join(collected).strip()
        if not content:
            yield _sse({"type": "error", "detail": "Model bu doküman için özet üretemedi."})
            return

        try:
            _persist_summary(user_id, document_id, content)
        except Exception:
            # Özet ekranda duruyor; kaydedememek onu kullanıcıdan saklamayı
            # gerektirmiyor — sadece bir dahaki sefere yeniden üretilir.
            logger.exception("Özet kaydedilemedi (document_id=%s)", document_id)

        yield _sse({"type": "done", "filename": filename})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
