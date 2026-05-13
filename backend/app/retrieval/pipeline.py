from typing import List, Dict
import numpy as np

import os
from app.config import settings
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_search import reciprocal_rank_fusion
from app.retrieval.reranker import Reranker
from app.ingestion.document_store import DocumentStore


class RetrievalPipeline:
    def __init__(self):
        if settings.hf_token:
            os.environ["HF_TOKEN"] = settings.hf_token
        self.embedder = EmbeddingModel(settings.embedding_model, settings.embedding_device)
        self.vector_store = VectorStore(settings.qdrant_path, settings.qdrant_collection, self.embedder.dim)
        self.bm25 = BM25Retriever()
        self.reranker = Reranker(settings.cross_encoder_model)
        self.doc_store = DocumentStore()

    def ingest_document(self, filename: str, text: str) -> str:
        from app.ingestion.text_splitter import chunk_text

        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        doc_id = self.doc_store.add_document(filename, chunks)

        enriched_chunks = []
        for i, c in enumerate(chunks):
            enriched_chunks.append({
                **c,
                "doc_id": doc_id,
                "filename": filename,
            })

        embeddings = self.embedder.encode([c["text"] for c in enriched_chunks])
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=hash(c["text"]) % (2**63),
                vector=embeddings[i].tolist(),
                payload={
                    "text": c["text"],
                    "doc_id": c["doc_id"],
                    "filename": c["filename"],
                    "chunk_index": c["index"],
                },
            )
            for i, c in enumerate(enriched_chunks)
        ]
        self.vector_store.upsert(points)

        all_chunks = self.doc_store.get_all_chunks()
        self.bm25.index(all_chunks)

        return doc_id

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        if top_k is None:
            top_k = settings.top_k_retrieval

        query_vec = self.embedder.encode_query(query)
        vector_results = self.vector_store.search(query_vec, top_k=top_k)

        bm25_results = self.bm25.search(query, top_k=top_k)

        fused = reciprocal_rank_fusion(
            vector_results, bm25_results,
            alpha=settings.hybrid_search_alpha,
            top_k=top_k,
        )

        reranked = self.reranker.rerank(query, fused, top_k=settings.top_k_rerank)
        return reranked

    def answer(self, query: str) -> Dict:
        contexts = self.search(query)
        from app.generation.llm_generator import LLMGenerator

        generator = LLMGenerator(
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            openai_api_key=settings.openai_api_key,
            anthropic_api_key=settings.anthropic_api_key,
            nvidia_api_key=settings.nvidia_api_key,
            nvidia_base_url=settings.nvidia_base_url,
        )
        result = generator.generate(query, contexts)
        result["contexts"] = contexts
        return result

    def get_documents(self) -> List[Dict]:
        return self.doc_store.get_all_documents()

    def remove_document(self, doc_id: str) -> bool:
        return self.doc_store.remove_document(doc_id)
