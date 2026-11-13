import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


class DocumentStore:
    def __init__(self, path: str = "./document_store.json"):
        self.path = path
        self._docs: dict[str, dict] = {}
        self._load()

    def _load(self):
        p = Path(self.path)
        if p.exists():
            data = p.read_text(encoding="utf-8").strip()
            if data:
                documents = json.loads(data)
                if not isinstance(documents, dict):
                    raise ValueError("Document store must contain a JSON object")
                self._docs = documents

    def _save(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._docs, f, indent=2)

    def _compute_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def add_document(self, filename: str, chunks: list[dict]) -> str:
        uploaded_at = datetime.now(timezone.utc).isoformat()  # noqa: UP017 - preserve Python 3.10 compatibility.
        doc_id = self._compute_hash(filename + uploaded_at)
        self._docs[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "uploaded_at": uploaded_at,
        }
        self._save()
        return doc_id

    def get_document(self, doc_id: str) -> dict | None:
        return self._docs.get(doc_id)

    def get_all_documents(self) -> list[dict]:
        return list(self._docs.values())

    def remove_document(self, doc_id: str) -> bool:
        if doc_id in self._docs:
            del self._docs[doc_id]
            self._save()
            return True
        return False

    def get_all_chunks(self) -> list[dict]:
        all_chunks = []
        for doc in self._docs.values():
            for chunk in doc["chunks"]:
                all_chunks.append({
                    **chunk,
                    "doc_id": doc["doc_id"],
                    "filename": doc["filename"],
                })
        return all_chunks
