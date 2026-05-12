# Production RAG System

Enterprise-grade document Q&A system with hybrid search, reranking, and evaluation pipelines.

## Architecture

```
User → Streamlit Frontend → FastAPI Backend → Embeddings → Vector DB (Qdrant)
                                              → BM25 Retrieval
                                              → Hybrid Search (RRF)
                                              → Cross-Encoder Reranking
                                              → LLM Generation (OpenAI/Anthropic)
                                              → Evaluation Metrics
```

## Features

- **Multi-format ingestion**: PDF, DOCX, TXT, MD
- **Hybrid search**: Semantic (sentence embeddings) + Keyword (BM25) with RRF fusion
- **Reranking**: Cross-encoder for precision
- **LLM generation**: OpenAI GPT or Anthropic Claude with source citations
- **Evaluation**: ROUGE, BERTScore, faithfulness metrics
- **CI/CD**: GitHub Actions + Docker

## Quick Start

```bash
# Clone & navigate
cd rag-system

# Backend
cd backend
pip install -r requirements.txt
cp ../.env.example .env  # Add your API keys
uvicorn app.main:app --reload

# Frontend (another terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Upload & ingest document |
| POST | `/query` | Ask a question |
| GET | `/documents` | List ingested documents |
| DELETE | `/documents/{id}` | Remove a document |
| POST | `/evaluate` | Evaluate answer quality |

## Docker

```bash
docker compose up --build
```

## Deployment

Deploy to Render using `render.yaml` or via the Render Dashboard.
