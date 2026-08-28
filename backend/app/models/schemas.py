"""API için Pydantic şemaları."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class UserCreate(BaseModel):
    username: str
    password: str
    preferred_language: str = "tr"


class UserOut(BaseModel):
    id: int
    username: str
    preferred_language: str

    class Config:
        from_attributes = True


class DocumentGroupCreate(BaseModel):
    name: str


class DocumentGroupOut(BaseModel):
    id: int
    name: str
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentGroupAssign(BaseModel):
    # None -> dokümanı grupsuz bırak.
    group_id: int | None = None


class DocumentOut(BaseModel):
    id: int
    filename: str
    file_type: str
    language: str | None
    uploaded_at: datetime
    group_id: int | None = None

    class Config:
        from_attributes = True


class AskRequest(BaseModel):
    question: str
    document_ids: list[int] | None = None  # None/boş -> tüm kullanıcı belgeleri
    language: str | None = None  # None -> otomatik algıla


class SourceRef(BaseModel):
    document_id: int
    filename: str
    chunk_index: int
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    # Hybrid RAG yeterince ilgili bir parça bulabildi mi? False ise cevap
    # kullanıcının dosyalarından değil, modelin genel bilgisinden geliyor
    # (bkz. api/ask.py:_retrieve ve services/llm.py:build_prompt).
    has_context: bool = True


class SummaryOut(BaseModel):
    document_id: int
    filename: str
    content: str
    model_alias: str | None = None
    updated_at: datetime

    class Config:
        from_attributes = True


class SummaryStatus(BaseModel):
    """Bir dokümanın özeti var mı? (Quiz ekranı, "özetten üret" seçeneğini
    aktif edip etmeyeceğine buna bakarak karar veriyor.)"""

    document_id: int
    has_summary: bool


class QuizRequest(BaseModel):
    document_id: int
    num_questions: int = 5
    # Quiz hangi metinden üretilsin?
    #   "auto"    -> özet varsa ondan, yoksa ham parçalardan (VARSAYILAN)
    #   "summary" -> yalnızca özetten; özet yoksa 400 döner
    #   "chunks"  -> her zaman ham doküman parçalarından
    # Özetten üretmek belirgin biçimde daha HIZLI: özet zaten damıtılmış ve
    # kısa olduğu için modelin okuması gereken context küçülüyor (bkz.
    # api/quiz.py:_load_quiz_context).
    source: Literal["auto", "summary", "chunks"] = "auto"


class QuizSource(BaseModel):
    """Quiz'in gerçekte hangi metinden üretildiği (arayüzde gösteriliyor)."""

    used: Literal["summary", "chunks"]


class QuizQuestionOut(BaseModel):
    question: str
    options: list[str] | None
    answer: str


class QuizOut(BaseModel):
    title: str
    questions: list[QuizQuestionOut]
    # Quiz'in hangi metinden üretildiği; arayüz "özetten üretildi" rozetini
    # buna bakarak gösteriyor.
    source: Literal["summary", "chunks"] = "chunks"


class CodeExplainRequest(BaseModel):
    document_id: int


class CodeExplainResponse(BaseModel):
    explanation: str
