from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from analysis.llm_client import generate_with_groq


CHAT_ROUTE_KEYWORDS: List[Tuple[str, Sequence[str]]] = [
    ("method", ("optimizer", "training", "learning rate", "batch size", "loss", "architecture", "implementation", "how was", "model was")),
    ("dataset", ("dataset", "datasets", "data", "benchmark", "corpus", "train set", "validation set", "test set")),
    ("limitations", ("limitation", "limitations", "weakness", "weaknesses", "challenge", "challenges", "failure case", "failure cases")),
    ("future_work", ("future work", "future direction", "next step", "open problem", "research gap", "gaps")),
    ("equations", ("equation", "equations", "formula", "formulas", "loss function", "mathematical", "derivation", "proof")),
    ("implementation", ("figure", "fig.", "diagram", "pipeline", "implementation", "code", "architecture", "figure 3")),
    ("results", ("result", "results", "performance", "accuracy", "f1", "ablation", "experiment", "evaluation")),
    ("contributions", ("contribution", "contributions", "novel", "novelty", "main idea", "what is new", "summary")),
    ("summary", ("what is this paper", "overview", "summary", "about", "main point")),
]


SECTION_TO_FALLBACK = {
    "method": ["method", "implementation", "summary"],
    "dataset": ["dataset", "method", "results"],
    "limitations": ["limitations", "summary", "future_work"],
    "future_work": ["future_work", "limitations", "summary"],
    "equations": ["equations", "method", "implementation"],
    "implementation": ["implementation", "method", "results"],
    "results": ["results", "summary", "method"],
    "contributions": ["contributions", "summary", "method"],
    "summary": ["summary", "contributions", "method"],
}


def normalize_section_name(section_name: str) -> str:
    lowered = str(section_name or "").lower().strip()
    for canonical in SECTION_TO_FALLBACK:
        if canonical in lowered:
            return canonical
    return lowered or "summary"


def route_question(question: str) -> List[str]:
    lowered = re.sub(r"\s+", " ", question.strip().lower())
    for section_name, keywords in CHAT_ROUTE_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return SECTION_TO_FALLBACK.get(section_name, [section_name])[:2]
    return ["summary", "method"]


def _section_text_from_paper(paper: Dict, section_name: str) -> str:
    for key in (section_name, section_name.replace("_", " "), section_name.replace("_", "")):
        value = paper.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_chunk_pages(chunk: Dict) -> str:
    pages = chunk.get("pages") or []
    if not pages:
        return "?"
    return ", ".join(str(page) for page in pages)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text or "").lower())


def _question_overlap_score(question: str, text: str) -> float:
    question_tokens = set(_tokenize(question))
    if not question_tokens:
        return 0.0
    text_tokens = set(_tokenize(text))
    if not text_tokens:
        return 0.0
    return len(question_tokens & text_tokens) / max(1, len(question_tokens))


def _collect_section_evidence(
    paper: Dict,
    question: str,
    routed_sections: Sequence[str],
    max_chunks: int = 2,
) -> Tuple[List[Dict], List[str]]:
    debug_report = paper.get("debug_report") or {}
    traces = debug_report.get("section_traces") or {}

    evidence_chunks: List[Dict] = []
    used_sections: List[str] = []
    candidates: List[Tuple[float, Dict]] = []

    for section_name in routed_sections:
        trace = traces.get(section_name) or traces.get(normalize_section_name(section_name))
        if not isinstance(trace, dict):
            section_text = _section_text_from_paper(paper, section_name)
            if section_text:
                evidence_chunks.append(
                    {
                        "section": section_name,
                        "page": paper.get("paper_info", {}).get("pages", "?"),
                        "chunk_id": None,
                        "parent_id": None,
                        "text": section_text[:1000],
                    }
                )
                used_sections.append(section_name)
            if len(evidence_chunks) >= max_chunks:
                break
            continue

        chunks = trace.get("chunks_sent_to_groq") or []
        for chunk in chunks:
            text = str(chunk.get("child_text") or chunk.get("parent_text") or "")
            score = _question_overlap_score(question, text)
            candidates.append(
                (
                    score,
                    {
                        "section": str(chunk.get("section") or section_name),
                        "page": _extract_chunk_pages(chunk),
                        "chunk_id": chunk.get("child_id"),
                        "parent_id": chunk.get("parent_id"),
                        "text": text[:1000],
                    },
                )
            )

    if candidates:
        for _, chunk in sorted(candidates, key=lambda item: item[0], reverse=True)[:max_chunks]:
            evidence_chunks.append(chunk)
            used_sections.append(str(chunk.get("section") or ""))

    return evidence_chunks, used_sections


def _build_chat_prompt(question: str, evidence_chunks: Sequence[Dict], paper: Dict) -> str:
    citations = []
    context_lines = []
    for index, chunk in enumerate(evidence_chunks, start=1):
        citations.append(
            f"[{index}] section={chunk['section']} page={chunk['page']} chunk_id={chunk.get('chunk_id')} parent_id={chunk.get('parent_id')}"
        )
        context_lines.append(
            f"[{index}] {chunk['text']}"
        )

    paper_info = paper.get("paper_info") or {}
    title = paper_info.get("title", "Unknown title")

    return f"""
You are answering a question about a research paper.
Use only the evidence provided.
Answer only the user's question directly.
Do not summarize the paper or restate whole sections.
Keep the answer concise, specific, and useful.
If the evidence is insufficient, say exactly what is missing.

Paper title: {title}
Question: {question}

Evidence citations:
{chr(10).join(citations)}

Evidence snippets:
{chr(10).join(context_lines)}
""".strip()


def answer_question(question: str, paper: Dict) -> Dict[str, object]:
    routed_sections = route_question(question)
    evidence_chunks, used_sections = _collect_section_evidence(paper, question, routed_sections, max_chunks=2)

    if not evidence_chunks:
        return {
            "answer": "I could not find enough evidence in the analyzed paper to answer that question.",
            "citations": [],
            "routed_sections": routed_sections,
            "used_sections": [],
        }

    prompt = _build_chat_prompt(question, evidence_chunks, paper)
    try:
        answer = generate_with_groq(prompt, max_tokens=320)
    except Exception:
        answer = ""

    if not isinstance(answer, str) or not answer.strip():
        answer = "I could not generate a reliable answer from the retrieved evidence."

    citations = [
        {
            "section": chunk["section"],
            "page": chunk["page"],
            "chunk_id": chunk.get("chunk_id"),
            "parent_id": chunk.get("parent_id"),
        }
        for chunk in evidence_chunks
    ]

    return {
        "answer": answer.strip(),
        "citations": citations,
        "routed_sections": routed_sections,
        "used_sections": used_sections,
    }
