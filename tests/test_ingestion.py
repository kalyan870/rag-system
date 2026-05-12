import pytest
from pathlib import Path
import tempfile
import json

from backend.app.ingestion.pdf_processor import extract_text_from_txt
from backend.app.ingestion.text_splitter import chunk_text
from backend.app.ingestion.document_store import DocumentStore


def test_extract_txt():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello world test content")
        path = f.name
    text = extract_text_from_txt(path)
    assert "Hello world test content" in text
    Path(path).unlink()


def test_chunk_text():
    text = "word " * 1000
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all("text" in c and "index" in c for c in chunks)


def test_document_store():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        store_path = f.name

    store = DocumentStore(store_path)
    chunks = [{"text": "chunk1", "index": 0}, {"text": "chunk2", "index": 1}]
    doc_id = store.add_document("test.txt", chunks)

    doc = store.get_document(doc_id)
    assert doc is not None
    assert doc["filename"] == "test.txt"
    assert doc["chunk_count"] == 2

    all_chunks = store.get_all_chunks()
    assert len(all_chunks) == 2

    store.remove_document(doc_id)
    assert store.get_document(doc_id) is None

    Path(store_path).unlink()
