from __future__ import annotations

import re


NO_EQUATIONS_MESSAGE = "No explicit mathematical equations found in paper."
NO_EVIDENCE_MESSAGE = "No supporting evidence found in the paper."

FORBIDDEN_TRAILINGS = ("-", "is", "are", "the", "a")


def _looks_like_complete_sentence(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith((".", "!", "?", ")", "]", "\"", "'")):
        return True
    return False


def validate_section_output(section_name: str, text):
    if isinstance(text, dict):
        return False, "generation_failed"

    if not isinstance(text, str):
        return False, "invalid_type"

    stripped = text.strip()

    if not stripped:
        return False, "empty"

    if section_name == "equations" and stripped == NO_EQUATIONS_MESSAGE:
        return True, "ok"

    if len(stripped) < 100:
        return False, "too_short"

    lowered = stripped.lower()
    if "groq api returned an empty response" in lowered:
        return False, "empty_response_message"
    if "error" in lowered:
        return False, "contains_error"

    if not _looks_like_complete_sentence(stripped):
        return False, "unfinished_sentence"

    trailing_tokens = re.sub(r"[^\w]+$", "", stripped).split()
    if not trailing_tokens:
        return False, "empty_trailing_token"
    trailing = trailing_tokens[-1].lower()
    if trailing in FORBIDDEN_TRAILINGS:
        return False, "bad_trailing_fragment"

    return True, "ok"


def validate_section_payload(section_name: str, payload):
    if not isinstance(payload, dict):
        return False, "invalid_payload"

    answer = payload.get("answer", "")
    status = payload.get("status")

    if status == "insufficient_evidence" and answer == NO_EVIDENCE_MESSAGE:
        return True, "ok"

    return validate_section_output(section_name, answer)


def build_failure_fallback(section_name: str, retriever):
    trace = getattr(retriever, "last_trace", None) or {}
    chunks = trace.get("chunks_sent_to_groq") or []
    if section_name == "equations":
        return NO_EQUATIONS_MESSAGE
    if chunks:
        snippet = " ".join(
            chunk.get("child_text", "")
            for chunk in chunks[:2]
        ).strip()
        if snippet:
            fallback = (
                "Generation was unstable, so here is the strongest retrieved evidence from the paper: "
                f"{snippet}"
            )
            if len(fallback) >= 100:
                return fallback
    return (
        "Generation was unstable, so the retrieved context is not sufficient for a reliable section. "
        "Please review the paper text and rerun the analysis after improving the retrieval signal."
    )
