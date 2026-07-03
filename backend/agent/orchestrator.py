from analysis.summary_agent import generate_summary
from analysis.contributions_agent import generate_contributions
from analysis.method_agent import generate_method_explanation
from analysis.dataset_agent import generate_dataset_explanation
from analysis.limitations_agent import generate_limitations
from analysis.future_work_agent import generate_future_work
from analysis.equation_agent import generate_equation_explanation
from analysis.implementation_agent import generate_implementation_ideas
from analysis.research_gap_agent import generate_research_gaps
from evaluation.metrics import evaluate_section_payload


SECTION_GENERATORS = [
    ("summary", generate_summary),
    ("contributions", generate_contributions),
    ("method", generate_method_explanation),
    ("dataset", generate_dataset_explanation),
    ("limitations", generate_limitations),
    ("future_work", generate_future_work),
    ("equations", generate_equation_explanation),
    ("implementation", generate_implementation_ideas),
    ("research_gaps", generate_research_gaps),
]


PUBLIC_SECTION_ORDER = [section_name for section_name, _ in SECTION_GENERATORS]


def _build_debug_report(retriever, section_results):
    # Keep the full debug payload assembled internally, but only expose it when debug mode is enabled.
    section_debug = {}
    fallback_invocations = []
    hallucination_blocks = []
    evaluation_report = {}

    for section_name, payload in section_results.items():
        retrieval_debug = payload.get("retrieval_debug") or {}
        section_debug[section_name] = {
            "query": retrieval_debug.get("query"),
            "retrieved_ids": retrieval_debug.get("retrieved_ids", []),
            "retrieved_chunk_ids": retrieval_debug.get("retrieved_chunk_ids", []),
            "retrieved_sections": retrieval_debug.get("retrieved_sections", []),
            "context_length": retrieval_debug.get("context_length", 0),
            "faiss_scores": retrieval_debug.get("faiss_scores", []),
            "bm25_scores": retrieval_debug.get("bm25_scores", []),
            "rerank_scores": retrieval_debug.get("rerank_scores", []),
            "final_context_length": retrieval_debug.get("final_context_length", 0),
            "status": retrieval_debug.get("status"),
            "fallback_mode": retrieval_debug.get("fallback_mode"),
            "chunks_sent_to_groq": retrieval_debug.get("chunks_sent_to_groq", []),
        }
        evaluation_report[section_name] = evaluate_section_payload(
            section_name=section_name,
            payload=payload,
            query_text=str(retrieval_debug.get("query_text", "")),
            expected_sections=retrieval_debug.get("allowed_sections") or [],
        )
        fallback_invocations.extend(retrieval_debug.get("fallback_invocations", []))
        if payload.get("status") in {"insufficient_evidence", "generation_failed"}:
            hallucination_blocks.append(
                {
                    "section": section_name,
                    "status": payload.get("status"),
                    "answer": payload.get("answer"),
                }
            )

    document_debug = getattr(retriever, "section_debug", {}) or {}

    return {
        "detected_sections": document_debug.get("headings", []),
        "section_coverage": document_debug.get("coverage", 0.0),
        "missing_sections": document_debug.get("missing_sections", []),
        "unmatched_headings": document_debug.get("unmatched_headings", []),
        "fallback_invocations": fallback_invocations,
        "section_traces": section_debug,
        "evaluation_report": evaluation_report,
        "hallucination_blocks": hallucination_blocks,
    }


def _build_public_report(section_results):
    # Strip every internal field so production responses stay compact and user-friendly.
    public_report = {}
    for section_name in PUBLIC_SECTION_ORDER:
        payload = section_results.get(section_name, {})
        public_report[section_name] = payload.get("answer") if isinstance(payload, dict) else payload
    return public_report


def analyze_paper(retriever, include_debug: bool = False):
    report = {}

    for section_name, generator in SECTION_GENERATORS:
        report[section_name] = generator(retriever)

    # Preserve the complete analysis internally so debugging data remains available without polluting production output.
    debug_report = _build_debug_report(retriever, report)
    report["debug_report"] = debug_report
    setattr(retriever, "last_analysis_report", report)
    setattr(retriever, "last_debug_report", debug_report)

    if include_debug:
        return report

    return _build_public_report(report)
