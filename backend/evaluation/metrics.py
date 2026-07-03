from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _normalize_tokens(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def _content_tokens(text: str) -> Set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "with",
        "we",
        "our",
        "they",
        "these",
        "those",
        "paper",
        "study",
        "method",
        "approach",
    }
    return {token for token in _normalize_tokens(text) if token not in stopwords and len(token) > 2}


def _cosine_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    left_counter = Counter(left)
    right_counter = Counter(right)
    if not left_counter or not right_counter:
        return 0.0

    shared_tokens = set(left_counter) & set(right_counter)
    numerator = sum(left_counter[token] * right_counter[token] for token in shared_tokens)
    left_norm = math.sqrt(sum(value * value for value in left_counter.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counter.values()))
    if not left_norm or not right_norm:
        return 0.0
    return float(numerator / (left_norm * right_norm))


def recall_at_k(retrieved_ids: Iterable[int], relevant_ids: Iterable[int], k: int) -> float:
    retrieved = list(retrieved_ids)[:k]
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hits = sum(1 for item in retrieved if item in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved_ids: Iterable[int], relevant_ids: Iterable[int]) -> float:
    relevant = set(relevant_ids)
    for index, item in enumerate(retrieved_ids, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def context_precision(retrieved_ids: Iterable[int], relevant_ids: Iterable[int]) -> float:
    retrieved = list(retrieved_ids)
    relevant = set(relevant_ids)
    if not retrieved:
        return 0.0
    hits = sum(1 for item in retrieved if item in relevant)
    return hits / len(retrieved)


def proxy_context_precision(
    retrieved_sections: Iterable[str],
    expected_sections: Optional[Iterable[str]] = None,
) -> float:
    retrieved = [str(section).lower().strip() for section in retrieved_sections if str(section).strip()]
    if not retrieved:
        return 0.0

    if expected_sections:
        expected = {str(section).lower().strip() for section in expected_sections if str(section).strip()}
        if expected:
            hits = sum(1 for section in retrieved if section in expected)
            return hits / len(retrieved)

    unique_sections = {section for section in retrieved if section}
    return len(unique_sections) / len(retrieved)


def answer_relevance(answer: str, query_text: str) -> float:
    answer_tokens = _content_tokens(answer)
    query_tokens = _content_tokens(query_text)
    if not answer_tokens or not query_tokens:
        return 0.0
    return len(answer_tokens & query_tokens) / len(query_tokens)


def faithfulness(answer: str, context: str) -> float:
    answer = answer or ""
    context = context or ""
    if not answer.strip() or not context.strip():
        return 0.0

    context_sentences = [
        sentence.strip()
        for sentence in SENTENCE_RE.split(context)
        if sentence and sentence.strip()
    ]
    if not context_sentences:
        return 0.0

    supported = 0
    answer_sentences = [
        sentence.strip()
        for sentence in SENTENCE_RE.split(answer)
        if sentence and sentence.strip()
    ]
    if not answer_sentences:
        return 0.0

    for sentence in answer_sentences:
        sentence_tokens = _content_tokens(sentence)
        if not sentence_tokens:
            continue

        best_overlap = 0.0
        for context_sentence in context_sentences:
            context_tokens = _content_tokens(context_sentence)
            if not context_tokens:
                continue
            overlap = len(sentence_tokens & context_tokens) / len(sentence_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
        if best_overlap >= 0.35:
            supported += 1

    return supported / max(len(answer_sentences), 1)


def generation_metric_stub(answer: str, context: str) -> Dict[str, Optional[float]]:
    return {
        "faithfulness": faithfulness(answer, context),
        "answer_relevance": answer_relevance(answer, context),
        "context_relevance": 1.0 if len((context or "").strip()) > 50 else 0.0,
    }


def build_ragas_compatible_report(
    *,
    retrieved_ids: Iterable[int],
    relevant_ids: Iterable[int],
    answer: str,
    context: str,
    query_text: str = "",
    retrieved_sections: Optional[Iterable[str]] = None,
    expected_sections: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    retrieved_ids = list(retrieved_ids)
    relevant_ids = list(relevant_ids)

    retrieval_report = {
        "recall_at_5": recall_at_k(retrieved_ids, relevant_ids, 5),
        "recall_at_10": recall_at_k(retrieved_ids, relevant_ids, 10),
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids),
        "context_precision": context_precision(retrieved_ids, relevant_ids),
    }

    if not relevant_ids and retrieved_sections is not None:
        retrieval_report["context_precision"] = proxy_context_precision(
            retrieved_sections,
            expected_sections=expected_sections,
        )

    generation_report = generation_metric_stub(answer, context)
    generation_report["answer_relevance"] = answer_relevance(answer, query_text or context)
    generation_report["faithfulness"] = faithfulness(answer, context)

    return {
        "retrieval": retrieval_report,
        "generation": generation_report,
        "mode": "gold" if relevant_ids else "proxy",
    }


def evaluate_section_payload(
    *,
    section_name: str,
    payload: Dict[str, object],
    query_text: str = "",
    expected_sections: Optional[Iterable[str]] = None,
    relevant_ids: Optional[Iterable[int]] = None,
) -> Dict[str, object]:
    retrieval_debug = payload.get("retrieval_debug") or {}
    context = retrieval_debug.get("groq_context") or payload.get("answer", "") or ""
    retrieved_ids = retrieval_debug.get("retrieved_chunk_ids") or []
    retrieved_sections = retrieval_debug.get("retrieved_sections") or []
    expected_sections = list(expected_sections or retrieval_debug.get("allowed_sections") or [])
    relevant_ids = list(relevant_ids or [])

    report = build_ragas_compatible_report(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        answer=str(payload.get("answer", "")),
        context=context,
        query_text=query_text or str(retrieval_debug.get("query_text", "")),
        retrieved_sections=retrieved_sections,
        expected_sections=expected_sections,
    )
    report["section"] = section_name
    report["status"] = payload.get("status")
    report["retrieved_chunk_ids"] = retrieved_ids
    report["retrieved_sections"] = retrieved_sections
    report["expected_sections"] = expected_sections
    return report
