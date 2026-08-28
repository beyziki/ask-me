"""Dosya yükleme sonrası metin çıkarma (extraction) ve chunking pipeline'ı.

Desteklenen tipler:
- PDF (.pdf)
- Markdown (.md)
- Kod dosyaları (.py, .c, .cpp, .java, .js, .ts, .go, .rs, vb.)
"""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from backend.app.core.config import settings

CODE_EXTENSIONS = {
    ".py", ".c", ".cpp", ".h", ".hpp", ".java", ".js", ".ts",
    ".go", ".rs", ".cs", ".rb", ".php", ".sql", ".sh",
}


def detect_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in {".md", ".markdown"}:
        return "md"
    if ext in CODE_EXTENSIONS:
        return "code"
    return "text"


def extract_text(path: Path, file_type: str) -> str:
    if file_type == "pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    # md, code ve text dosyaları için doğrudan oku
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(
    text: str,
    chunk_size: int = settings.chunk_size,
    overlap: int = settings.chunk_overlap,
) -> list[str]:
    """Basit kelime bazlı sliding-window chunking.

    Not: Kod dosyaları için ileride fonksiyon/blok bazlı chunking'e
    (örn. tree-sitter ile) geçilebilir; MVP aşamasında kelime bazlı yeterli.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


def process_upload(path: Path, filename: str) -> tuple[str, list[str]]:
    """Dosyayı işleyip (file_type, chunk_listesi) döner."""
    file_type = detect_file_type(filename)
    text = extract_text(path, file_type)
    chunks = chunk_text(text)
    return file_type, chunks
