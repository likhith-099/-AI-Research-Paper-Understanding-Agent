from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def build_retrieval_trace(
    query_name,
    query_text,
    section_filter,
    allowed_sections,
    fallback_mode,
    fallback_invocations,
    dense_results,
    sparse_results,
    reranked_results,
    selected_chunks,
    prompt_context,
    status,
):
    faiss_scores = []
    bm25_scores = []
    rerank_scores = []

    for chunk, distance, original_index in dense_results:
        faiss_scores.append(
            {
                "chunk_id": chunk["child_id"],
                "section": chunk["section"],
                "score": float(1.0 / (1.0 + distance)),
                "distance": float(distance),
                "chunk_index": original_index,
            }
        )

    for chunk, score in sparse_results:
        bm25_scores.append(
            {
                "chunk_id": chunk["child_id"],
                "section": chunk["section"],
                "score": float(score),
            }
        )

    for item in reranked_results:
        rerank_scores.append(
            {
                "chunk_id": item["chunk"]["child_id"],
                "section": item["chunk"]["section"],
                "score": float(item["score"]),
            }
        )

    trace = {
        "query": query_name or query_text,
        "query_text": query_text,
        "section_filter": section_filter or [],
        "allowed_sections": allowed_sections or [],
        "fallback_mode": fallback_mode,
        "fallback_invocations": fallback_invocations,
        "retrieved_chunks": [
            {
                "chunk_id": chunk["child_id"],
                "parent_id": chunk["parent_id"],
                "section": chunk["section"],
                "score_context": chunk.get("section_confidence", 0.0),
            }
            for chunk in selected_chunks
        ],
        "retrieved_ids": [chunk["child_id"] for chunk in selected_chunks],
        "retrieved_sections": [chunk["section"] for chunk in selected_chunks],
        "retrieved_chunk_ids": [chunk["child_id"] for chunk in selected_chunks],
        "faiss_scores": faiss_scores,
        "bm25_scores": bm25_scores,
        "rerank_scores": rerank_scores,
        "chunks_sent_to_groq": selected_chunks,
        "groq_context": prompt_context,
        "context_length": len(prompt_context or ""),
        "final_context_length": len(prompt_context or ""),
        "status": status,
    }

    logger.info(
        "Retrieval trace query=%s fallback_mode=%s status=%s chunk_ids=%s context_len=%s",
        trace["query"],
        fallback_mode,
        status,
        trace["retrieved_chunk_ids"],
        trace["final_context_length"],
    )

    return trace
