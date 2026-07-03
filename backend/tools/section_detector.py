from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


CANONICAL_SECTIONS = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "literature review",
    "method",
    "methodology",
    "approach",
    "framework",
    "experimental setup",
    "dataset",
    "datasets",
    "data",
    "experiments",
    "results",
    "discussion",
    "limitations",
    "future work",
    "conclusion",
    "references",
]


SECTION_SYNONYMS = {
    "abstract": ["abstract", "executive summary", "summary"],
    "introduction": ["introduction", "intro", "overview", "motivation"],
    "background": ["background", "preliminaries", "preliminary", "foundations"],
    "related work": [
        "related work",
        "related works",
        "related studies",
        "literature review",
        "literature survey",
        "prior work",
        "prior art",
        "state of the art",
        "survey",
        "survey of literature",
    ],
    "literature review": ["literature review"],
    "method": [
        "method",
        "methods",
        "methodology",
        "approach",
        "model",
        "architecture",
        "proposed method",
        "system design",
    ],
    "methodology": ["methodology", "proposed methodology"],
    "approach": ["approach", "proposed approach"],
    "framework": ["framework", "pipeline", "system architecture"],
    "experimental setup": [
        "experimental setup",
        "experiment setup",
        "setup",
        "experiments setup",
        "implementation details",
    ],
    "dataset": ["dataset", "benchmark", "corpus"],
    "datasets": ["datasets"],
    "data": ["data"],
    "experiments": [
        "experiments",
        "evaluation",
        "experimentation",
        "empirical evaluation",
        "benchmark results",
    ],
    "results": ["results", "findings", "performance", "comparative results"],
    "discussion": ["discussion", "analysis", "implications", "case study", "error analysis"],
    "limitations": [
        "limitations",
        "limitation",
        "weaknesses",
        "threats to validity",
        "challenges",
        "open challenges",
    ],
    "future work": [
        "future work",
        "future directions",
        "future research",
        "open problems",
        "research directions",
        "future studies",
    ],
    "conclusion": [
        "conclusion",
        "conclusions",
        "closing remarks",
        "final remarks",
        "summary and conclusion",
        "conclusion and future work",
    ],
    "references": ["references", "bibliography"],
}


SURVEY_HINTS = {
    "related work": ["taxonomy", "survey", "survey paper", "related studies", "comparison of"],
    "background": ["overview", "preliminaries", "foundations", "motivation"],
    "method": ["framework", "pipeline", "method", "model", "architecture", "algorithm", "design"],
    "experiments": ["evaluation", "benchmark", "comparison", "experimental", "experiments", "results"],
    "results": ["performance", "findings", "quantitative", "qualitative", "comparison"],
    "discussion": ["discussion", "analysis", "applications", "case study", "implications"],
    "limitations": ["limitations", "challenges", "open issues", "threats", "weaknesses"],
    "future work": ["future work", "future directions", "future research", "open problems"],
    "conclusion": ["conclusion", "summary", "closing remarks", "final thoughts"],
}


HEADING_PREFIX_RE = re.compile(
    r"^\s*(?:section|chapter|appendix)?\s*(?:\d+(?:\.\d+)*|[IVXLCDM]+|[A-Z])(?:[\.\)\-:]+|\s+)+",
    re.IGNORECASE,
)

HEADING_LIKE_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*|[IVXLCDM]+|[A-Z])?[\.\)\-: ]*\s*[A-Za-z][A-Za-z0-9 ,&/\-\(\)]{1,90}\s*$"
)


def _normalize_heading(line: str) -> str:
    candidate = re.sub(r"\s+", " ", line).strip()
    candidate = HEADING_PREFIX_RE.sub("", candidate).strip()
    candidate = re.sub(r"[\s:;\-\u2013\u2014]+$", "", candidate).strip()
    return candidate


def _score_heading(normalized: str, alias: str) -> float:
    if normalized == alias:
        return 1.0
    if normalized.startswith(alias + " "):
        return 0.92
    if alias in normalized and len(normalized) <= len(alias) + 24:
        return 0.84
    return 0.0


def _match_canonical_heading(line: str) -> Optional[Tuple[str, float]]:
    stripped = _normalize_heading(line)
    normalized = stripped.lower()
    if not normalized:
        return None

    word_count = len(normalized.split())
    if word_count > 9:
        return None
    if any(token in normalized for token in [". ", "? ", "! "]):
        return None
    if normalized.endswith((".", "?", "!", ";", ",")) and normalized not in {"i.", "ii.", "iii.", "iv.", "v."}:
        return None

    best_match = None
    best_score = 0.0

    for canonical, aliases in SECTION_SYNONYMS.items():
        for alias in aliases:
            score = _score_heading(normalized, alias)
            if score > best_score:
                best_match = (canonical, score)
                best_score = score

    if best_match:
        return best_match

    for canonical, hints in SURVEY_HINTS.items():
        if any(hint in normalized for hint in hints):
            return canonical, 0.74

    return None


def _is_heading_like(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 110:
        return False
    if HEADING_LIKE_RE.match(stripped):
        return True
    return False


def detect_sections_verbose(text: str) -> Dict[str, object]:
    """
    Return a structured section map with confidence, coverage, and unmatched headings.
    """
    lines = text.splitlines()
    headings: List[Dict[str, object]] = []
    unmatched_headings: List[Dict[str, object]] = []

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        matched = _match_canonical_heading(stripped)
        if matched:
            canonical, confidence = matched
            headings.append(
                {
                    "index": index,
                    "section": canonical,
                    "heading": stripped,
                    "confidence": confidence,
                }
            )
            continue

        if _is_heading_like(stripped):
            unmatched_headings.append(
                {
                    "index": index,
                    "heading": stripped,
                }
            )
            logger.warning("Unmatched heading candidate: %s", stripped)

    headings.sort(key=lambda item: item["index"])

    sections: Dict[str, Dict[str, object]] = {}
    if not headings:
        sections["full paper"] = {
            "start": 0,
            "end": len(text),
            "content": text.strip(),
            "order": 0,
            "confidence": 0.0,
            "heading": "full paper",
            "canonical_section": "full paper",
            "section_label": "full paper",
        }
        return {
            "sections": sections,
            "headings": headings,
            "unmatched_headings": unmatched_headings,
            "coverage": 0.0,
            "missing_sections": CANONICAL_SECTIONS,
        }

    canonical_counts: Dict[str, int] = {}

    for position, heading in enumerate(headings):
        start_line = heading["index"]
        end_line = headings[position + 1]["index"] if position + 1 < len(headings) else len(lines)

        section_lines = lines[start_line + 1:end_line]
        content = "\n".join(section_lines).strip()
        start_pos = sum(len(lines[i]) + 1 for i in range(start_line))
        end_pos = sum(len(lines[i]) + 1 for i in range(end_line)) if end_line < len(lines) else len(text)
        canonical_section = heading["section"]
        canonical_counts[canonical_section] = canonical_counts.get(canonical_section, 0) + 1
        section_label = canonical_section
        if canonical_counts[canonical_section] > 1:
            section_label = f"{canonical_section} ({canonical_counts[canonical_section]})"
        section_key = f"{section_label} [{position}]"

        sections[section_key] = {
            "start": start_pos,
            "end": end_pos,
            "content": content,
            "order": position,
            "confidence": heading["confidence"],
            "heading": heading["heading"],
            "canonical_section": canonical_section,
            "section_label": section_label,
        }

    detected = {
        info.get("canonical_section", section_name)
        for section_name, info in sections.items()
    }
    expected = {
        section
        for section in CANONICAL_SECTIONS
        if section not in {"dataset", "datasets", "data", "method", "methodology", "approach", "framework"}
    }
    coverage = len(detected & expected) / max(len(expected), 1)

    return {
        "sections": sections,
        "headings": headings,
        "unmatched_headings": unmatched_headings,
        "coverage": coverage,
        "missing_sections": sorted(expected - detected),
    }


def detect_sections(text: str):
    """
    Backwards-compatible helper returning just section content.
    """
    verbose = detect_sections_verbose(text)
    sections = verbose["sections"]

    result = {}
    for section_name, info in sections.items():
        canonical_section = info.get("canonical_section", section_name)
        if canonical_section in result and result[canonical_section].strip():
            result[canonical_section] = "\n\n".join(
                part for part in [result[canonical_section], info["content"]] if part.strip()
            ).strip()
        else:
            result[canonical_section] = info["content"]
    return result
