from sentence_transformers import CrossEncoder
from typing import List, Dict


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        if not results:
            return []

        pairs = [(query, r["text"]) for r in results]
        scores = self.model.predict(pairs)

        for i, r in enumerate(results):
            r["rerank_score"] = float(scores[i])

        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_k]
