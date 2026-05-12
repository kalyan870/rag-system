from typing import List, Dict
import numpy as np


def reciprocal_rank_fusion(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    alpha: float = 0.5,
    top_k: int = 20,
) -> List[Dict]:
    seen = {}
    k = 60

    for rank, r in enumerate(vector_results):
        idx = r.get("id") or hash(r["text"])
        score = 1.0 / (k + rank + 1)
        if idx not in seen:
            seen[idx] = {**r, "vector_score": r.get("score", 0), "bm25_score": 0.0, "fusion_score": 0.0}
        else:
            seen[idx]["vector_score"] = r.get("score", 0)
        seen[idx]["fusion_score"] += alpha * score

    for rank, r in enumerate(bm25_results):
        idx = r.get("id") or hash(r["text"])
        score = 1.0 / (k + rank + 1)
        fusion_add = (1 - alpha) * score
        if idx not in seen:
            seen[idx] = {**r, "vector_score": 0.0, "bm25_score": r.get("bm25_score", 0), "fusion_score": 0.0}
        else:
            seen[idx]["bm25_score"] = r.get("bm25_score", 0)
        seen[idx]["fusion_score"] += fusion_add

    fused = list(seen.values())
    fused.sort(key=lambda x: x["fusion_score"], reverse=True)
    return fused[:top_k]
