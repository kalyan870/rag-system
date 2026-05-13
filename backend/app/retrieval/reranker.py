import requests
from typing import List, Dict, Optional
import os


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        self.model_name = model_name
        self.hf_token = os.getenv("HF_TOKEN")
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self._local_model = None
        self._init_local()

    def _init_local(self):
        try:
            from sentence_transformers import CrossEncoder
            self._local_model = CrossEncoder(self.model_name)
        except:
            self._local_model = None

    def _call_api(self, pairs: List[List[str]]) -> List[float]:
        headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        response = requests.post(self.api_url, headers=headers, json={"inputs": pairs})
        if response.status_code == 200:
            return [r["score"] if isinstance(r, dict) else r for r in response.json()]
        raise RuntimeError(f"Reranker API error: {response.status_code}")

    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        if not results:
            return []

        pairs = [[query, r["text"]] for r in results]

        if self._local_model:
            scores = self._local_model.predict(pairs)
        else:
            try:
                scores = self._call_api(pairs)
            except Exception as e:
                for i, r in enumerate(reversed(sorted(results, key=lambda x: x.get("fusion_score", x.get("score", 0))))):
                    r["rerank_score"] = float(len(results) - i)
                results.sort(key=lambda x: x["rerank_score"], reverse=True)
                return results[:top_k]

        for i, r in enumerate(results):
            r["rerank_score"] = float(scores[i])

        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_k]
