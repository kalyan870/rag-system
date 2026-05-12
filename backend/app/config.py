from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    app_name: str = "RAG System"
    debug: bool = False

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_dimension: int = 384

    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    chunk_size: int = 512
    chunk_overlap: int = 64

    top_k_retrieval: int = 20
    top_k_rerank: int = 5

    hybrid_search_alpha: float = 0.5

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-haiku-20240307"

    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    qdrant_path: str = "./qdrant_data"
    qdrant_collection: str = "documents"

    upload_dir: str = "./uploads"

    max_file_size_mb: int = 50
    allowed_extensions: str = ".pdf,.docx,.txt,.md"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
