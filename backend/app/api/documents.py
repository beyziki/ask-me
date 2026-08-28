"""Dosya yükleme, listeleme, silme ve gruplama uçları."""
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, get_current_user_id
from backend.app.core.config import settings
from backend.app.db.models import Chunk, Document, DocumentGroup, Quiz
from backend.app.models.schemas import DocumentGroupAssign, DocumentGroupCreate, DocumentGroupOut, DocumentOut
from backend.app.services.ingestion import process_upload
from backend.app.services.rag import add_chunks_to_index, drop_rows_from_index, rebuild_index

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


def _get_owned_group(group_id: int, user_id: int, db: Session) -> DocumentGroup:
    group = db.query(DocumentGroup).get(group_id)
    if not group or group.owner_id != user_id:
        raise HTTPException(status_code=404, detail="Grup bulunamadı")
    return group


@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    file: UploadFile = File(...),
    # Yükleme sırasında isteğe bağlı olarak doğrudan bir gruba atamak için;
    # dosya listesinden sonradan da değiştirilebilir (bkz. assign_document_group).
    group_id: int | None = Form(None),
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    if group_id is not None:
        _get_owned_group(group_id, user_id, db)

    user_dir = settings.uploads_dir / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    dest = user_dir / file.filename
    dest.write_bytes(await file.read())

    file_type, chunks = process_upload(dest, file.filename)

    document = Document(
        owner_id=user_id,
        filename=file.filename,
        file_type=file_type,
        storage_path=str(dest),
        group_id=group_id,
    )
    db.add(document)
    db.flush()  # document.id almak için

    vector_rows = add_chunks_to_index(user_id, chunks)
    for idx, (content, vrow) in enumerate(zip(chunks, vector_rows)):
        db.add(
            Chunk(
                document_id=document.id,
                owner_id=user_id,
                chunk_index=idx,
                content=content,
                vector_row=vrow,
            )
        )

    db.commit()
    db.refresh(document)
    return document


@router.get("", response_model=list[DocumentOut])
def list_documents(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    return db.query(Document).filter(Document.owner_id == user_id).all()


@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Yanlışlıkla eklenen ya da artık istenmeyen bir dosyayı siler.

    Doküman (ve `cascade="all, delete-orphan"` sayesinde onun chunk'ları,
    bkz. db/models.py:Document.chunks) DB'den siliniyor, diskteki dosya
    kaldırılıyor. FAISS tarafı biraz daha dikkat istiyor: düz `IndexFlatIP`
    tek tek satır silmeyi desteklemediği için index'i baştan kurup kalan
    chunk'ların `vector_row` değerlerini güncelliyoruz — aksi halde satır
    numaraları kayar ve semantic search yanlış chunk'ları döner.

    Bunu MEVCUT vektörleri kopyalayarak yapıyoruz (bkz.
    rag.py:drop_rows_from_index), yeniden embed'leyerek DEĞİL. Eskiden burada
    `rebuild_index` çağrılıyordu: kalan HER chunk sıfırdan embed'leniyor,
    yüzlerce chunk'ta silme işlemi CPU'da dakikalarca sürüyordu (gözlemlenen
    davranış: "silme butonu bir türlü bitmiyor"). Vektörler zaten index
    dosyasında durduğu için bu iş tamamen gereksizdi. Hızlı yol
    uygulanamazsa `None` döner ve güvenli tarafta kalmak için eski yola
    düşüyoruz.
    """
    document = db.query(Document).get(document_id)
    if not document or document.owner_id != user_id:
        raise HTTPException(status_code=404, detail="Doküman bulunamadı")

    remaining_chunks = (
        db.query(Chunk)
        .filter(Chunk.owner_id == user_id, Chunk.document_id != document_id)
        .order_by(Chunk.id)
        .all()
    )
    new_rows = drop_rows_from_index(user_id, [c.vector_row for c in remaining_chunks])
    if new_rows is None:
        logger.info(
            "Index hızlı yoldan güncellenemedi, yeniden embed'leniyor (user_id=%s)", user_id
        )
        new_rows = rebuild_index(user_id, [c.content for c in remaining_chunks])
    for chunk, row in zip(remaining_chunks, new_rows):
        chunk.vector_row = row

    # Bu dokümandan üretilmiş quiz'ler (bkz. db/models.py:Quiz.document_id)
    # SİLİNMİYOR (kullanıcının çözüm geçmişi kalıcı, frontend zaten dosya
    # adını o an localStorage'a kaydetmişti) — sadece artık var olmayan bir
    # dokümana sarkan referansı temizliyoruz.
    db.query(Quiz).filter(Quiz.document_id == document_id).update({Quiz.document_id: None})

    storage_path = Path(document.storage_path)
    db.delete(document)
    db.commit()

    try:
        if storage_path.exists():
            storage_path.unlink()
    except OSError:
        # Diskteki dosyayı silemesek bile DB tarafı zaten tutarlı; bunu
        # kullanıcıya 500 olarak yansıtmak yerine loglayıp geçiyoruz.
        logger.warning("Doküman dosyası diskten silinemedi: %s", storage_path)


@router.post("/groups", response_model=DocumentGroupOut)
def create_document_group(
    payload: DocumentGroupCreate,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Grup adı boş olamaz")
    group = DocumentGroup(owner_id=user_id, name=name)
    db.add(group)
    db.commit()
    db.refresh(group)
    return group


@router.get("/groups", response_model=list[DocumentGroupOut])
def list_document_groups(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    return db.query(DocumentGroup).filter(DocumentGroup.owner_id == user_id).all()


@router.delete("/groups/{group_id}", status_code=204)
def delete_document_group(
    group_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    group = _get_owned_group(group_id, user_id, db)
    # Grup silindiğinde içindeki dosyalar SİLİNMİYOR, yalnızca "Grupsuz"a
    # dönüyor (bkz. db/models.py:Document.group_id notu).
    db.query(Document).filter(Document.group_id == group_id).update({Document.group_id: None})
    db.delete(group)
    db.commit()


@router.patch("/{document_id}/group", response_model=DocumentOut)
def assign_document_group(
    document_id: int,
    payload: DocumentGroupAssign,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    document = db.query(Document).get(document_id)
    if not document or document.owner_id != user_id:
        raise HTTPException(status_code=404, detail="Doküman bulunamadı")
    if payload.group_id is not None:
        _get_owned_group(payload.group_id, user_id, db)
    document.group_id = payload.group_id
    db.commit()
    db.refresh(document)
    return document
