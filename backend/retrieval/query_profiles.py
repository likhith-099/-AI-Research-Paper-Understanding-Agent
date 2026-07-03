from __future__ import annotations

from copy import deepcopy


SECTION_FALLBACKS = {
    "summary": [
        ["abstract", "introduction", "conclusion"],
        ["background", "related work", "literature review"],
        [],
    ],
    "contributions": [
        ["abstract", "introduction", "related work", "conclusion"],
        ["background", "literature review"],
        [],
    ],
    "method": [
        ["method", "methodology", "approach", "framework", "experimental setup"],
        ["experiments", "results"],
        [],
    ],
    "dataset": [
        ["dataset", "datasets", "data", "experimental setup", "experiments"],
        ["method", "methodology", "approach"],
        [],
    ],
    "limitations": [
        ["limitations", "discussion", "conclusion"],
        ["future work", "related work"],
        [],
    ],
    "future_work": [
        ["future work", "discussion", "conclusion"],
        ["limitations", "results"],
        [],
    ],
    "equations": [
        ["method", "methodology", "approach", "framework", "experimental setup"],
        ["experiments", "results"],
        [],
    ],
    "implementation": [
        ["method", "methodology", "approach", "framework", "experimental setup"],
        ["experiments"],
        [],
    ],
    "research_gaps": [
        ["limitations", "discussion", "future work", "conclusion", "related work"],
        ["method", "methodology"],
        [],
    ],
}


QUERY_OPTIONS = {
    "equations": {
        "equation_only": True,
        "min_chunk_count": 1,
        "min_context_length": 200,
    },
    "summary": {
        "equation_only": False,
        "min_chunk_count": 2,
        "min_context_length": 500,
    },
    "contributions": {
        "equation_only": False,
        "min_chunk_count": 2,
        "min_context_length": 500,
    },
    "dataset": {
        "equation_only": False,
        "min_chunk_count": 2,
        "min_context_length": 500,
    },
    "limitations": {
        "equation_only": False,
        "min_chunk_count": 2,
        "min_context_length": 500,
    },
    "future_work": {
        "equation_only": False,
        "min_chunk_count": 2,
        "min_context_length": 500,
    },
    "method": {
        "equation_only": False,
        "min_chunk_count": 2,
        "min_context_length": 500,
    },
    "implementation": {
        "equation_only": False,
        "min_chunk_count": 2,
        "min_context_length": 500,
    },
    "research_gaps": {
        "equation_only": False,
        "min_chunk_count": 2,
        "min_context_length": 500,
    },
}


def _normalize_sections(sections):
    return [
        section.lower().strip()
        for section in sections
        if section and section.strip()
    ]


def get_query_profile(query_name=None, section_filter=None, available_sections=None):
    profile = {
        "fallback_groups": deepcopy(SECTION_FALLBACKS.get(query_name or "", [[]])),
        "equation_only": QUERY_OPTIONS.get(query_name or "", {}).get("equation_only", False),
        "min_chunk_count": QUERY_OPTIONS.get(query_name or "", {}).get("min_chunk_count", 2),
        "min_context_length": QUERY_OPTIONS.get(query_name or "", {}).get("min_context_length", 500),
    }

    if section_filter:
        filter_set = set(_normalize_sections(section_filter))
        filtered_groups = []
        for group in profile["fallback_groups"]:
            mapped = [section for section in group if section in filter_set]
            if mapped:
                filtered_groups.append(mapped)
        if filtered_groups:
            filtered_groups.append([])
            profile["fallback_groups"] = filtered_groups

    if available_sections is not None:
        available_set = set(_normalize_sections(available_sections))
        normalized_groups = []
        for group in profile["fallback_groups"]:
            mapped = [section for section in group if section in available_set]
            if mapped:
                normalized_groups.append(mapped)
        normalized_groups.append([])
        profile["fallback_groups"] = normalized_groups

    return profile
