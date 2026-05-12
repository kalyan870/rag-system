from rank_bm25 import BM25Okapi
from typing import List, Dict
import nltk
import re

nltk.download("punkt_tab", quiet=True)


class BM25Retriever:
    def __init__(self):
        self.bm25: BM25Okapi = None
        self.chunk_data: List[Dict] = []

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return nltk.word_tokenize(text)

    def index(self, chunks: List[Dict]):
        tokenized = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.chunk_data = chunks

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        if not self.bm25:
            return []
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        scored = [
            {**self.chunk_data[i], "bm25_score": float(scores[i])}
            for i in range(len(self.chunk_data))
        ]
        scored.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored[:top_k]

    def clear(self):
        self.bm25 = None
        self.chunk_data = []
