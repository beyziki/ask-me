"""Otomatik quiz üretimi uç noktası."""
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, get_current_user_id
from backend.app.db.base import SessionLocal
from backend.app.db.models import Chunk, Document, Quiz, QuizQuestion, Summary, User
from backend.app.models.schemas import QuizOut, QuizQuestionOut, QuizRequest
from backend.app.services.llm import FoundryNotAvailable, GpuContextLost, raise_if_gpu_context_lost
from backend.app.services.quiz import _parse_quiz_json, generate_quiz, generate_quiz_stream

router = APIRouter(prefix="/quiz", tags=["quiz"])
logger = logging.getLogger(__name__)

# Quiz için modele gönderilecek en fazla parça sayısı. `/ask` sorguya özel
# olarak (hibrit RAG ile) sadece en alakalı ~7-8 parçayı seçiyor, ama quiz'in
# soracağı belirli bir soru olmadığı için önceden TÜM doküman parçaları
# context olarak gönderiliyordu. Büyük bir PDF (ör. 50+ sayfalık ders notu)
# yüzlerce parçaya bölünebiliyor; bunların hepsini tek istekte göndermek
# hem üretimi (bazen 10+ dakikaya kadar) inanılmaz yavaşlatıyor hem de
# modelin context penceresini zorluyordu. Bu yüzden dokümanın tamamını
# temsil edecek şekilde (baştan sona eşit aralıklarla) sabit sayıda parça
# örnekliyoruz. 15 -> 10: 8GB VRAM'li kartlarda (ör. RTX 4060 Laptop) quiz'in
# daha büyük context'i + max_tokens=1400 çıktısı, KV cache için modelin
# zaten neredeyse belleği doldurmuş olmasıyla birleşince "CudaMallocArray
# / MatMulNBits ... out of memory" hatasına yol açabiliyordu. 10 -> 7: quiz
# artık streaming ile (bkz. `create_quiz_stream`) üretim sırasında canlı bir
# ilerleme sinyali gösterse de, TOPLAM süreyi de gerçekten kısaltmak için
# context'i daha da küçülttük — prefill (context'in işlenmesi) süresi parça
# sayısıyla orantılı büyüyor. Kapsam biraz daha daralıyor ama dokümanın
# tamamını temsil eden örnekleme mantığı (aşağıya bak) aynı kalıyor.
MAX_QUIZ_CHUNKS = 7


def _load_quiz_context(
    payload: QuizRequest, user_id: int, db: Session
) -> tuple[int, str, list[str], str, str]:
    """Quiz için context'i hazırlar; `/quiz` ve `/quiz/stream` tarafından
    ortak kullanılan adım.

    Dönüş: (document_id, filename, context, language, used_source).

    KAYNAK SEÇİMİ (bkz. QuizRequest.source):
    Doküman için bir ÖZET varsa varsayılan olarak ondan üretiyoruz. Bunun
    nedeni hız: özet zaten damıtılmış ve kısa bir metin, oysa ham parçalar
    (örneklenmiş olsa bile) çok daha uzun. Prefill süresi context uzunluğuyla
    doğru orantılı olduğu için özetten üretim belirgin biçimde daha hızlı
    bitiyor. Ayrıca özet dokümanın TAMAMINDAN üretildiği için (bkz.
    services/summary.py map-reduce) kapsamı da aşağıdaki örneklemeden daha
    iyi: örnekleme dokümanın büyük kısmını atlıyor.

    Özet yoksa eski davranışa (ham parçalardan örnekleme) düşülüyor.
    """
    document = db.query(Document).filter(
        Document.id == payload.document_id, Document.owner_id == user_id
    ).first()
    if not document:
        raise HTTPException(status_code=404, detail="Doküman bulunamadı")

    summary = None
    if payload.source in ("auto", "summary"):
        summary = (
            db.query(Summary).filter(Summary.document_id == document.id).first()
        )
        if summary is None and payload.source == "summary":
            raise HTTPException(
                status_code=400,
                detail="Bu doküman için henüz özet üretilmedi. Önce Özet ekranından özet çıkar.",
            )

    if summary is not None:
        used_source = "summary"
        context = [summary.content]
    else:
        used_source = "chunks"
        chunks = (
            db.query(Chunk)
            .filter(Chunk.document_id == document.id)
            .order_by(Chunk.chunk_index)
            .all()
        )
        if not chunks:
            raise HTTPException(status_code=400, detail="Bu dokümanda işlenmiş içerik yok")

        if len(chunks) > MAX_QUIZ_CHUNKS:
            step = len(chunks) / MAX_QUIZ_CHUNKS
            chunks = [chunks[int(i * step)] for i in range(MAX_QUIZ_CHUNKS)]
        context = [c.content for c in chunks]

    user = db.query(User).get(user_id)
    # ORM nesnelerinin (document, user) alanlarını session kapanmadan ÖNCE
    # düz değerlere kopyalıyoruz — aksi halde session kapandıktan sonra
    # (LLM çağrısı bittikten dakikalar sonra) bu alanlara erişmek "detached
    # instance" hatası riski taşır.
    document_id = document.id
    filename = document.filename
    language = user.preferred_language

    # İhtiyacımız olan tüm veriyi yukarıda çektik; aşağıdaki LLM çağrısı
    # (Foundry Local) birkaç dakikaya kadar sürebiliyor. Bu süre boyunca
    # session'ı (ve dolayısıyla SQLite'ın okuma kilidini/bağlantı havuzundan
    # bir bağlantıyı) açık tutmamak için burada kapatıyoruz — aksi halde
    # SQLite'ta bekleyen bir yazma işlemi (ör. login) bu kilit boşalana kadar
    # bloke oluyordu (bkz. ask.py'deki aynı düzeltme).
    db.close()

    return document_id, filename, context, language, used_source


def _persist_quiz(
    user_id: int,
    document_id: int,
    filename: str,
    raw_questions: list[dict],
    used_source: str = "chunks",
) -> QuizOut:
    """Üretilen soruları DB'ye kaydeder ve `QuizOut` olarak döner.

    Yazma işlemleri için kısa ömürlü, ayrı bir session kullanıyor (bkz.
    `_load_quiz_context`'teki aynı not: uzun süren LLM çağrısı sırasında ana
    session'ı açık tutmamak için).
    """
    quiz_title = f"{filename} - Quiz"
    fresh_db = SessionLocal()
    try:
        quiz = Quiz(owner_id=user_id, document_id=document_id, title=quiz_title)
        fresh_db.add(quiz)
        fresh_db.flush()

        questions_out = []
        for q in raw_questions:
            fresh_db.add(
                QuizQuestion(
                    quiz_id=quiz.id,
                    question=q["question"],
                    options=json.dumps(q.get("options")) if q.get("options") else None,
                    answer=q["answer"],
                )
            )
            questions_out.append(QuizQuestionOut(**q))

        fresh_db.commit()
    finally:
        fresh_db.close()

    # `quiz_title`'ı fresh_db kapanmadan önce yakalanan yerel değişkenden
    # kullanıyoruz — commit sonrası ORM nesnesinin alanları expire olduğu
    # için `quiz.title`'a session kapandıktan sonra erişmek hataya yol açar.
    return QuizOut(title=quiz_title, questions=questions_out, source=used_source)


@router.post("", response_model=QuizOut)
def create_quiz(
    payload: QuizRequest,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Tek seferde (streaming olmayan) quiz üretimi. Programatik kullanım ve
    testler için korunuyor; asıl arayüz artık `/quiz/stream`'i kullanıyor
    (bkz. aşağıdaki `create_quiz_stream`)."""
    document_id, filename, context, language, used_source = _load_quiz_context(
        payload, user_id, db
    )
    raw_questions = generate_quiz(context, payload.num_questions, language)
    return _persist_quiz(user_id, document_id, filename, raw_questions, used_source)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/stream")
def create_quiz_stream(
    payload: QuizRequest,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Quiz'i üretim sırasında canlı bir ilerleme sinyaliyle (SSE) üretir.

    Model çıktısı ham JSON metni olduğu ve parça parça geçerli bir JSON
    olarak render edilemeyeceği için `token` olayları SORULARI DEĞİL, ham
    metin parçalarını taşır — frontend bunları bir ilerleme göstergesi
    (ör. "N karakter alındı") için kullanır (bkz.
    frontend-web/src/pages/QuizPage.tsx). Akış bitince metin ayrıştırılıp
    DB'ye kaydedilir ve nihai sorular TEK SEFERDE bir `result` olayıyla
    gönderilir. Olay sırası: (birden çok) `token` -> `result` (ya da bir
    hata olursa `error`).
    """
    document_id, filename, context, language, used_source = _load_quiz_context(
        payload, user_id, db
    )

    def event_stream():
        # Arayüz "özetten üretiliyor" rozetini üretim BAŞLAMADAN gösterebilsin
        # diye kaynağı ilk olayla bildiriyoruz.
        yield _sse({"type": "source", "used": used_source})
        raw_parts: list[str] = []
        try:
            for piece in generate_quiz_stream(context, payload.num_questions, language):
                raw_parts.append(piece)
                yield _sse({"type": "token", "content": piece})
        except (FoundryNotAvailable, GpuContextLost) as exc:
            yield _sse({"type": "error", "detail": str(exc)})
            return
        except Exception as exc:
            # CUDA bağlamı bozulduysa bunu ham traceback yerine anlaşılır bir
            # mesaja çeviriyoruz (bkz. llm.py:GpuContextLost).
            try:
                raise_if_gpu_context_lost(exc)
            except GpuContextLost as fatal:
                logger.warning("Quiz üretimi GPU bağlam kaybıyla durdu: %s", fatal)
                yield _sse({"type": "error", "detail": str(fatal)})
                return
            logger.exception("/quiz/stream sırasında beklenmeyen hata")
            yield _sse(
                {"type": "error", "detail": "Quiz üretilirken beklenmeyen bir hata oluştu."}
            )
            return

        try:
            raw_questions = _parse_quiz_json("".join(raw_parts))
            result = _persist_quiz(
                user_id, document_id, filename, raw_questions, used_source
            )
        except Exception as exc:
            # Model bazen istenen JSON formatına tam uymayan bir çıktı
            # üretebiliyor (bozuk JSON, eksik alan vb.); bunu backend'i
            # çökertmek yerine kullanıcıya anlaşılır bir hata olarak gösteriyoruz.
            logger.warning("Quiz JSON ayrıştırılamadı/kaydedilemedi: %s", exc)
            yield _sse(
                {
                    "type": "error",
                    "detail": "Model geçerli bir quiz formatı üretmedi, lütfen tekrar dene.",
                }
            )
            return

        yield _sse({"type": "result", "quiz": result.model_dump()})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
