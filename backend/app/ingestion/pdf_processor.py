import pdfplumber
from docx import Document
from pathlib import Path
from typing import List
import os
from huggingface_hub import InferenceClient


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
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("File contains binary data. If this is an image, please use .png or .jpg extension for OCR support.")

    nulls = sum(1 for c in text if ord(c) == 0)
    if nulls > 0:
        raise ValueError("File contains binary data. If this is an image, please use .png or .jpg extension for OCR support.")

    nontext = sum(1 for c in text if not (c.isprintable() or c in "\n\r\t\f "))
    if nontext > len(text) * 0.1:
        raise ValueError("File appears to contain binary data. If this is an image, please use .png or .jpg extension for OCR support.")
    return text


def extract_text_from_image(path: str) -> str:
    token = os.getenv("HF_TOKEN")
    client = InferenceClient(token=token) if token else InferenceClient()
    try:
        result = client.image_to_text(path, model="microsoft/trocr-base-printed")
        text = result if isinstance(result, str) else result.get("generated_text", "")
        if not text.strip():
            result = client.image_to_text(path, model="nlpconnect/vit-gpt2-image-captioning")
            text = result if isinstance(result, str) else result.get("generated_text", "")
        if not text.strip():
            raise ValueError("No text could be read from the image. Ensure the image contains clear, readable text.")
        return text
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not read image. Ensure the image contains clear, readable text. Error: {e}")


def extract_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    elif ext in (".txt", ".md"):
        return extract_text_from_txt(path)
    elif ext in (".png", ".jpg", ".jpeg"):
        return extract_text_from_image(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
