import pytest
import numpy as np

from backend.app.retrieval.hybrid_search import reciprocal_rank_fusion
from backend.app.retrieval.reranker import Reranker


def test_reciprocal_rank_fusion():
    vector_results = [
        {"id": 1, "text": "doc a", "score": 0.9, "filename": "a.pdf"},
        {"id": 2, "text": "doc b", "score": 0.8, "filename": "b.pdf"},
    ]
    bm25_results = [
        {"id": 3, "text": "doc c", "bm25_score": 10.0, "filename": "c.pdf"},
        {"id": 2, "text": "doc b", "bm25_score": 9.0, "filename": "b.pdf"},
    ]
    fused = reciprocal_rank_fusion(vector_results, bm25_results, alpha=0.5, top_k=5)
    assert len(fused) == 3
    ids = [r["id"] for r in fused]
    assert 1 in ids
    assert 2 in ids
    assert 3 in ids
    assert fused[0]["id"] == 2


def test_fusion_empty():
    fused = reciprocal_rank_fusion([], [], top_k=5)
    assert fused == []


def test_reranker_initialization():
    reranker = Reranker("cross-encoder/ms-marco-MiniLM-L6-v2")
    assert reranker is not None
