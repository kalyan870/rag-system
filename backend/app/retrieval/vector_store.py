import numpy as np
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue


class VectorStore:
    def __init__(self, path: str = "./qdrant_data", collection: str = "documents", dim: int = 384):
        self.client = QdrantClient(path=path)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection for c in collections):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def upsert(self, points: List[PointStruct]):
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: np.ndarray, top_k: int = 20, doc_id: Optional[str] = None) -> List[Dict]:
        query_filter = None
        if doc_id:
            query_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            {
                "id": r.id,
                "score": r.score,
                "text": r.payload.get("text", ""),
                "doc_id": r.payload.get("doc_id", ""),
                "filename": r.payload.get("filename", ""),
                "chunk_index": r.payload.get("chunk_index", 0),
            }
            for r in results
        ]

    def delete_collection(self):
        self.client.delete_collection(collection_name=self.collection)

    def close(self):
        self.client.close()
