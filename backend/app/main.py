from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import os
import uuid

from app.config import settings
from app.retrieval.pipeline import RetrievalPipeline
from app.ingestion.pdf_processor import extract_text
from app.evaluation.metrics import evaluate_answer

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
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {settings.allowed_extensions}")

    temp_path = os.path.join(settings.upload_dir, f"{uuid.uuid4()}{ext}")
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        text = extract_text(temp_path)
        if not text.strip():
            raise HTTPException(400, "No text could be extracted from the file.")

        doc_id = pipeline.ingest_document(file.filename, text)
        return {"doc_id": doc_id, "filename": file.filename, "message": "Document ingested successfully"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/query")
def query_documents(query: str = Form(...)):
    if not query.strip():
        raise HTTPException(400, "Query cannot be empty.")
    result = pipeline.answer(query)
    return result


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
