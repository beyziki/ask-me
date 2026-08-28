"""SQLAlchemy engine ve session yönetimi."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from backend.app.core.config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    # Varsayılan pool_size=5 + max_overflow=10, LLM çağrısı (Foundry Local
    # soğuk başlangıç/üretim) uzun sürdüğünde ve kullanıcı bu sırada tekrar
    # istek gönderdiğinde hızla tükeniyordu (QueuePool ... timeout hatası).
    # Havuzu büyütmek tek başına kök nedeni çözmüyor (bkz. ask.py'deki
    # db.close() değişikliği) ama ek bir güvenlik payı sağlıyor.
    pool_size=10,
    max_overflow=20,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def run_startup_migrations() -> None:
    """`Base.metadata.create_all` yalnızca EKSİK TABLOLARI oluşturur; zaten
    var olan bir tabloya sonradan eklenen bir SÜTUNU otomatik eklemiyor. Bu
    projede Alembic gibi bir migration aracı yok (bkz. README), bu yüzden
    böyle küçük şema değişikliklerini burada elle, "sütun var mı kontrol et,
    yoksa ekle" şeklinde uyguluyoruz.

    Şu an tek migration: `documents.group_id` (bkz. db/models.py:DocumentGroup).
    Bu fonksiyon `Base.metadata.create_all()`'dan HEMEN SONRA çağrılmalı
    (bkz. main.py) — yeni bir kurulumda `documents` tablosu zaten
    `group_id` ile oluşturulacağı için orada yapılacak bir şey olmaz; bu
    yalnızca ask-me'yi bu özellikten ÖNCE kurmuş, verisi olan kullanıcılar
    için gerekli.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "documents" not in inspector.get_table_names():
        return  # yeni kurulum, create_all zaten doğru şemayla oluşturdu

    existing_columns = {col["name"] for col in inspector.get_columns("documents")}
    if "group_id" in existing_columns:
        return

    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE documents ADD COLUMN group_id INTEGER"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
