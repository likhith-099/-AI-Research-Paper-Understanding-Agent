from __future__ import annotations

from analysis.llm_client import generate_with_groq
from analysis.output_validation import (
    validate_section_output,
    NO_EVIDENCE_MESSAGE,
    build_failure_fallback,
)


QUERY_EXPANSIONS = {
    "summary": [
        "SUMMARY",
        "ABSTRACT",
        "OVERVIEW",
        "KEY FINDINGS",
    ],
    "contributions": [
        "CONTRIBUTIONS",
        "NOVELTY",
        "KEY FINDINGS",
        "MAIN IDEA",
    ],
    "dataset": [
        "DATASET",
        "CORPUS",
        "BENCHMARK",
        "TRAINING DATA",
    ],
    "limitations": [
        "LIMITATIONS",
        "WEAKNESSES",
        "FAILURE CASES",
        "CHALLENGES",
    ],
    "future_work": [
        "FUTURE WORK",
        "FUTURE DIRECTIONS",
        "OPEN PROBLEMS",
        "EXTENSIONS",
    ],
    "method": [
        "METHOD",
        "METHODOLOGY",
        "APPROACH",
        "FRAMEWORK",
    ],
    "implementation": [
        "IMPLEMENTATION",
        "REPRODUCTION",
        "TRAINING PIPELINE",
    ],
    "research_gaps": [
        "RESEARCH GAPS",
        "OPEN PROBLEMS",
        "UNEXPLORED AREAS",
        "LIMITATIONS",
    ],
    "equations": [
        "EQUATIONS",
        "FORMULAS",
        "LOSS FUNCTION",
        "MATHEMATICAL FORMULATION",
    ],
}


def _format_evidence_chunks(selected_chunks):
    return [
        {
            "chunk_id": chunk["child_id"],
            "parent_id": chunk["parent_id"],
            "section": chunk["section"],
        }
        for chunk in selected_chunks
    ]


def generate_evidence_based_section(
    retriever,
    *,
    section_name,
    query_text,
    prompt,
    max_tokens,
    top_k=4,
):
    semantic_suffix = "\n".join(QUERY_EXPANSIONS.get(section_name, []))
    expanded_query = f"{query_text}\n{semantic_suffix}".strip()
    bundle = retriever.retrieve_context_bundle(
        expanded_query,
        top_k=top_k,
        query_name=section_name,
    )

    selected_chunks = bundle["selected_chunks"]
    retrieval_debug = bundle["retrieval_trace"]
    evidence_chunks = _format_evidence_chunks(selected_chunks)
    source_sections = [
        chunk["section"]
        for chunk in selected_chunks
    ]

    if not bundle["context"].strip():
        return {
            "answer": NO_EVIDENCE_MESSAGE,
            "evidence_chunks": evidence_chunks,
            "source_sections": source_sections,
            "status": "insufficient_evidence",
            "retrieval_debug": retrieval_debug,
            "generation_debug": None,
        }

    evidence_block = "\n".join(
        f"- chunk_id={chunk['child_id']} section={chunk['section']} text={chunk['child_text']}"
        for chunk in selected_chunks
    )

    full_prompt = f"""
You are an AI research assistant.

Answer only using the evidence in the context. If the evidence is not sufficient, say so explicitly.
Do not invent facts.

Task:
{prompt}

Evidence chunks:
{evidence_block}

Context:
{bundle['context']}
"""

    generation_debug = None
    for _ in range(3):
        try:
            generated, generation_debug = generate_with_groq(
                full_prompt,
                max_tokens=max_tokens,
                return_metadata=True,
            )
        except RuntimeError as exc:
            generation_debug = {
                "error": str(exc),
            }
            continue

        if isinstance(generated, dict):
            continue

        is_valid, _ = validate_section_output(section_name, generated)
        if is_valid:
            return {
                "answer": generated,
                "evidence_chunks": evidence_chunks,
                "source_sections": source_sections,
                "status": "ok" if bundle.get("status") == "ok" else "ok_fallback",
                "retrieval_debug": retrieval_debug,
                "generation_debug": generation_debug,
            }

    fallback_answer = build_failure_fallback(section_name, retriever)
    return {
        "answer": fallback_answer,
        "evidence_chunks": evidence_chunks,
        "source_sections": source_sections,
        "status": "ok_fallback" if fallback_answer != NO_EVIDENCE_MESSAGE else "insufficient_evidence",
        "retrieval_debug": retrieval_debug,
        "generation_debug": generation_debug,
    }
