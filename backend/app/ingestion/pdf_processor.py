import pdfplumber
from docx import Document
from pathlib import Path
from typing import List


def extract_text_from_pdf(path: str) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text_from_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def extract_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    elif ext in (".txt", ".md"):
        return extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
