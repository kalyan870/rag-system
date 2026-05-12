from backend.app.generation.llm_generator import LLMGenerator


def test_generator_prompt_format():
    generator = LLMGenerator(provider="openai", model="gpt-4o-mini")
    contexts = [
        {"text": "The sky is blue.", "filename": "doc1.pdf"},
        {"text": "Water is wet.", "filename": "doc2.pdf"},
    ]
    prompt = generator._make_prompt("What color is the sky?", contexts)
    assert "What color is the sky?" in prompt
    assert "[Document 1]" in prompt
    assert "[Document 2]" in prompt
    assert "The sky is blue." in prompt
    assert "Water is wet." in prompt


def test_generator_empty_contexts():
    generator = LLMGenerator(provider="openai", model="gpt-4o-mini")
    result = generator.generate("test query", [])
    assert "No relevant documents" in result["answer"]
    assert result["citations"] == []


def test_fallback_generation():
    generator = LLMGenerator(provider="openai", model="gpt-4o-mini")
    contexts = [
        {"text": "Fallback test content.", "filename": "test.pdf", "rerank_score": 0.95},
    ]
    result = generator._fallback_generate("test query", contexts)
    assert "Fallback test content" in result
    assert "test query" in result
