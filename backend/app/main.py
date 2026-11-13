import os
import uuid
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.evaluation.metrics import evaluate_answer
from app.ingestion.pdf_processor import extract_text
from app.retrieval.pipeline import RetrievalPipeline

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RetrievalPipeline()

Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}


@app.post("/upload")
async def upload_document(file: Annotated[UploadFile, File()]):
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {settings.allowed_extensions}")

    temp_path = os.path.join(settings.upload_dir, f"{uuid.uuid4()}{ext}")
    try:
        async with aiofiles.open(temp_path, "wb") as f:
            await f.write(await file.read())
        text = extract_text(temp_path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, "Could not read file") from e

    if not text.strip():
        os.remove(temp_path)
        raise HTTPException(400, "No text could be extracted from the file. The file appears to contain images or binary data.")

    try:
        doc_id = pipeline.ingest_document(file.filename, text)
        return {"doc_id": doc_id, "filename": file.filename, "message": "Document ingested successfully"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/query")
def query_documents(query: str = Form(...)):
    if not query.strip():
        raise HTTPException(400, "Query cannot be empty.")
    try:
        result = pipeline.answer(query)
        return result
    except Exception as e:
        raise HTTPException(500, "Query failed. Please retry or contact support.") from e


@app.get("/documents")
def list_documents():
    docs = pipeline.get_documents()
    return {"documents": [{"doc_id": d["doc_id"], "filename": d["filename"], "chunk_count": d["chunk_count"], "uploaded_at": d["uploaded_at"]} for d in docs]}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if pipeline.remove_document(doc_id):
        return {"message": "Document removed"}
    raise HTTPException(404, "Document not found")


@app.post("/evaluate")
def evaluate(
    query: str = Form(...),
    reference_answer: str = Form(...),
    generated_answer: str = Form(...),
):
    result = pipeline.search(query)
    contexts = [c["text"] for c in result]
    metrics = evaluate_answer(query, reference_answer, generated_answer, contexts)
    return metrics
