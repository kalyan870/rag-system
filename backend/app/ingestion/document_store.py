from pathlib import Path
from typing import List, Dict, Optional
import json
import hashlib
from datetime import datetime


class DocumentStore:
    def __init__(self, path: str = "./document_store.json"):
        self.path = path
        self._docs: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        p = Path(self.path)
        if p.exists():
            with open(p, "r") as f:
                self._docs = json.load(f)

    def _save(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._docs, f, indent=2)

    def _compute_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def add_document(self, filename: str, chunks: List[Dict]) -> str:
        doc_id = self._compute_hash(filename + datetime.utcnow().isoformat())
        self._docs[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "uploaded_at": datetime.utcnow().isoformat(),
        }
        self._save()
        return doc_id

    def get_document(self, doc_id: str) -> Optional[Dict]:
        return self._docs.get(doc_id)

    def get_all_documents(self) -> List[Dict]:
        return list(self._docs.values())

    def remove_document(self, doc_id: str) -> bool:
        if doc_id in self._docs:
            del self._docs[doc_id]
            self._save()
            return True
        return False

    def get_all_chunks(self) -> List[Dict]:
        all_chunks = []
        for doc in self._docs.values():
            for chunk in doc["chunks"]:
                all_chunks.append({
                    **chunk,
                    "doc_id": doc["doc_id"],
                    "filename": doc["filename"],
                })
        return all_chunks
