from typing import List, Dict
import numpy as np

try:
    from rouge_score import rouge_scorer
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False

import nltk

nltk.download("punkt_tab", quiet=True)


def compute_rouge(reference: str, candidate: str) -> Dict[str, float]:
    if not _ROUGE_AVAILABLE:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {k: v.fmeasure for k, v in scores.items()}


def compute_bert_score(references: List[str], candidates: List[str]) -> Dict[str, float]:
    if not _BERTSCORE_AVAILABLE or not references or not candidates:
        return {"bert_score_f1": 0.0}
    P, R, F1 = bert_score(candidates, references, lang="en", verbose=False)
    return {"bert_score_f1": float(F1.mean())}


def compute_faithfulness(contexts: List[str], answer: str) -> float:
    answer_sentences = nltk.sent_tokenize(answer)
    if not answer_sentences:
        return 0.0

    supported = 0
    for sent in answer_sentences:
        sent_lower = sent.lower()
        for ctx in contexts:
            if any(phrase in ctx.lower() for phrase in sent_lower.split()[:5]):
                supported += 1
                break
    return supported / len(answer_sentences)


def evaluate_answer(
    query: str,
    reference_answer: str,
    generated_answer: str,
    contexts: List[str],
) -> Dict:
    metrics = {}

    rouge_scores = compute_rouge(reference_answer, generated_answer)
    metrics.update(rouge_scores)

    bert = compute_bert_score([reference_answer], [generated_answer])
    metrics.update(bert)

    metrics["faithfulness"] = compute_faithfulness(contexts, generated_answer)

    metrics["context_relevance"] = len(contexts) / max(len(contexts), 5)

    return metrics
