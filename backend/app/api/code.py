"""Kod dosyası analiz/açıklama uç noktası."""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, get_current_user_id
from backend.app.db.models import Document, User
from backend.app.models.schemas import CodeExplainRequest, CodeExplainResponse
from backend.app.services.code_analysis import explain_code

router = APIRouter(prefix="/code", tags=["code"])


@router.post("/explain", response_model=CodeExplainResponse)
def explain(
    payload: CodeExplainRequest,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    document = db.query(Document).filter(
        Document.id == payload.document_id, Document.owner_id == user_id
    ).first()
    if not document or document.file_type != "code":
        raise HTTPException(status_code=404, detail="Kod dosyası bulunamadı")

    code_text = Path(document.storage_path).read_text(encoding="utf-8", errors="ignore")
    user = db.query(User).get(user_id)
    explanation = explain_code(code_text, user.preferred_language)
    return CodeExplainResponse(explanation=explanation)
