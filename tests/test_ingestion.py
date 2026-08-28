"""Dosya tipi tespiti ve chunking pipeline'ı için birim testleri."""
from backend.app.services.ingestion import chunk_text, detect_file_type


def test_detect_file_type():
    assert detect_file_type("notes.pdf") == "pdf"
    assert detect_file_type("notes.md") == "md"
    assert detect_file_type("main.py") == "code"
    assert detect_file_type("solution.cpp") == "code"
    assert detect_file_type("readme.txt") == "text"


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"kelime{i}" for i in range(1, 51))  # 50 kelime
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    # ardışık chunk'lar arasında overlap kadar kelime tekrar etmeli
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-5:] == second_words[:5]


def test_chunk_text_handles_empty_input():
    assert chunk_text("") == []


def test_chunk_text_single_chunk_when_shorter_than_size():
    text = "sadece birkaç kelime"
    chunks = chunk_text(text, chunk_size=500, overlap=80)
    assert chunks == [text]
