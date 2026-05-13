import numpy as np
from typing import List
import os
from huggingface_hub import InferenceClient


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = f"sentence-transformers/{model_name}"
        self.dimension = 384
        token = os.getenv("HF_TOKEN")
        self.client = InferenceClient(token=token) if token else InferenceClient()

    def encode(self, texts: List[str]) -> np.ndarray:
        result = self.client.feature_extraction(texts, model=self.model_name)
        if isinstance(result, list):
            result = np.array(result, dtype=np.float32)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / norms

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]

    @property
    def dim(self) -> int:
        return self.dimension
