"""Veritabanı tabloları: kullanıcı, doküman, chunk, quiz."""
import datetime as dt

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey
)
from sqlalchemy.orm import relationship

from backend.app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    preferred_language = Column(String(8), default="tr")
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    document_groups = relationship(
        "DocumentGroup", back_populates="owner", cascade="all, delete-orphan"
    )
    quizzes = relationship("Quiz", back_populates="owner", cascade="all, delete-orphan")


class DocumentGroup(Base):
    """Kullanıcının dosyalarını organize etmek için kullandığı, serbestçe
    adlandırılabilir grup/klasör (ör. "Bilgisayar Ağları", "Veri Yapıları").
    Soru Sor ve Quiz ekranlarında bir grubun tamamını tek seferde seçmek
    için de kullanılıyor (bkz. frontend-web/src/pages/AskPage.tsx,
    QuizPage.tsx)."""

    __tablename__ = "document_groups"

    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    owner = relationship("User", back_populates="document_groups")
    documents = relationship("Document", back_populates="group")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(16), nullable=False)  # pdf | md | code
    language = Column(String(8), nullable=True)
    storage_path = Column(String(512), nullable=False)
    uploaded_at = Column(DateTime, default=dt.datetime.utcnow)
    # NULL -> gruplanmamış ("Grupsuz"). Grup silindiğinde de NULL'a dönüyor
    # (bkz. backend/app/api/documents.py:delete_document_group) — dosyanın
    # kendisi silinmiyor, sadece grup ilişkisi kalkıyor.
    group_id = Column(Integer, ForeignKey("document_groups.id"), nullable=True, index=True)

    owner = relationship("User", back_populates="documents")
    group = relationship("DocumentGroup", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    # Doküman silinince özeti de gitmeli (özet yalnızca o dokümana ait).
    summary = relationship(
        "Summary", back_populates="document", uselist=False, cascade="all, delete-orphan"
    )


class Chunk(Base):
    """Doküman parçaları — hem BM25 hem semantic index için kaynak."""
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    # FAISS içindeki vektör satırına referans (owner bazlı index içinde)
    vector_row = Column(Integer, nullable=True)

    document = relationship("Document", back_populates="chunks")


class Summary(Base):
    """Bir dokümanın tamamı için üretilmiş özet.

    Neden kalıcı: özet üretmek dokümanın TAMAMINI modelden geçirmeyi
    gerektiriyor (uzun dosyalarda parça parça, bkz.
    services/summary.py) — yani en pahalı işlemlerden biri. Bir kez üretip
    saklayınca hem tekrar açıldığında anında geliyor, hem de "özetten quiz
    üret" akışı (bkz. api/quiz.py) hazır metni kullanarak ham parçalardan
    üretmeye göre çok daha hızlı çalışıyor.

    Doküman başına EN FAZLA bir özet tutuluyor; yeniden üretildiğinde
    mevcut kayıt güncelleniyor (bkz. api/summary.py). Doküman silindiğinde
    özeti de siliniyor (`cascade`, bkz. Document.summary).
    """

    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    document_id = Column(
        Integer, ForeignKey("documents.id"), nullable=False, unique=True, index=True
    )
    content = Column(Text, nullable=False)
    # Özeti hangi modelin ürettiği: kullanıcı modeli değiştirdiğinde eski
    # özetin farklı bir modelden geldiğini arayüzde gösterebilmek için.
    model_alias = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

    document = relationship("Document", back_populates="summary")


class Quiz(Base):
    __tablename__ = "quizzes"

    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    owner = relationship("User", back_populates="quizzes")
    questions = relationship("QuizQuestion", back_populates="quiz", cascade="all, delete-orphan")


class QuizQuestion(Base):
    __tablename__ = "quiz_questions"

    id = Column(Integer, primary_key=True)
    quiz_id = Column(Integer, ForeignKey("quizzes.id"), nullable=False)
    question = Column(Text, nullable=False)
    options = Column(Text, nullable=True)  # JSON string olarak saklanır (çoktan seçmeli ise)
    answer = Column(Text, nullable=False)
    source_chunk_id = Column(Integer, ForeignKey("chunks.id"), nullable=True)

    quiz = relationship("Quiz", back_populates="questions")
