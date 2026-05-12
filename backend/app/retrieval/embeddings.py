from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]

    @property
    def dim(self) -> int:
        return self.dimension
