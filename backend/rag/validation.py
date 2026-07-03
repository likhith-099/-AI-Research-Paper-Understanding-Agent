from __future__ import annotations

import logging
from typing import Dict, List, Tuple


logger = logging.getLogger(__name__)


EXPECTED_SECTIONS = [
    "abstract",
    "introduction",
    "methodology",
    "dataset",
    "results",
    "conclusion",
]


def validate_chunk_metadata(chunks: List[Dict]) -> Dict[str, object]:
    observed_sections = []
    missing_sections = []

    for chunk in chunks:
        section = str(chunk.get("section", "")).strip()
        print(section)
        observed_sections.append(section)

    lower_sections = {section.lower() for section in observed_sections if section}
    for section in EXPECTED_SECTIONS:
        if section not in lower_sections:
            missing_sections.append(section)

    logger.info(
        "Chunk metadata sections observed=%s missing=%s",
        sorted(lower_sections),
        missing_sections,
    )

    return {
        "observed_sections": observed_sections,
        "missing_sections": missing_sections,
        "has_expected_sections": not missing_sections,
    }


def validate_parent_child_mapping(chunks: List[Dict]) -> Dict[str, object]:
    invalid_children = []

    for chunk in chunks:
        parent_id = chunk.get("parent_id")
        if "parent_id" not in chunk or parent_id is None:
            invalid_children.append(
                {
                    "child_id": chunk.get("child_id"),
                    "parent_id": parent_id,
                    "section": chunk.get("section"),
                }
            )

    if invalid_children:
        logger.warning("Invalid parent-child mapping detected: %s", invalid_children)
    else:
        logger.info("Parent-child mapping validated for %s chunks", len(chunks))

    return {
        "parent_count": len({chunk.get("parent_id") for chunk in chunks if chunk.get("parent_id") is not None}),
        "invalid_children": invalid_children,
        "valid": not invalid_children,
    }
