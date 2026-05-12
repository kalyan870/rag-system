from typing import List, Dict, Optional
from openai import OpenAI
import anthropic
import os


class LLMGenerator:
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.openai_client = None
        self.anthropic_client = None

        if provider == "openai":
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)

    def _make_prompt(self, query: str, contexts: List[Dict]) -> str:
        context_text = "\n\n".join(
            f"[Document {i+1}] (Source: {c['filename']}):\n{c['text']}"
            for i, c in enumerate(contexts)
        )

        return f"""You are a precise document Q&A assistant. Answer the user's question based solely on the provided document excerpts. If the documents do not contain enough information to answer, say so.

For every claim you make, cite the source document number in brackets like [1], [2], etc.

**Documents:**
{context_text}

**Question:** {query}

**Answer:**"""

    def generate(self, query: str, contexts: List[Dict]) -> Dict:
        if not contexts:
            return {
                "answer": "No relevant documents found to answer the question.",
                "citations": [],
            }

        prompt = self._make_prompt(query, contexts)

        if self.provider == "openai" and self.openai_client:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            answer = response.choices[0].message.content
        elif self.provider == "anthropic" and self.anthropic_client:
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.content[0].text
        else:
            answer = self._fallback_generate(query, contexts)

        citations = [
            {"text": c["text"], "filename": c["filename"], "relevance": c.get("rerank_score", c.get("score", 0))}
            for c in contexts
        ]

        return {"answer": answer, "citations": citations}

    def _fallback_generate(self, query: str, contexts: List[Dict]) -> str:
        context_text = "\n\n".join(
            f"From {c['filename']}: {c['text'][:300]}"
            for c in contexts[:3]
        )
        lines = [
            f"Based on the retrieved documents, here is the answer to: {query}",
            "",
            f"Context used:",
            context_text,
            "",
            "Note: LLM API key not configured. This is a fallback response showing the retrieved context.",
        ]
        return "\n".join(lines)
