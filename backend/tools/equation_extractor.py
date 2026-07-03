import re
from typing import List


EQUATION_PATTERNS = [
    r"[A-Za-z0-9\(\)\[\]\{\}\+\-\*/\^,\.\s]{0,80}(?:=|≈|≤|≥)[A-Za-z0-9\(\)\[\]\{\}\+\-\*/\^,\.\s]{2,180}",
    r"[A-Za-z0-9\(\)\[\]\{\}\+\-\*/\^,\.\s]{0,80}(?:Σ|Π|∇|λ|α|β|γ)[A-Za-z0-9\(\)\[\]\{\}\+\-\*/\^,\.\s]{0,180}",
    r"[^.\n]{0,40}(?:loss|objective|equation|formula)[^.\n]{0,180}",
]


def _clean_equation(candidate: str) -> str:
    cleaned = re.sub(r"\s+", " ", candidate).strip(" :-;,.")
    return cleaned


def extract_equations(text: str, limit: int = 20) -> List[str]:
    matches = []

    for pattern in EQUATION_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidate = _clean_equation(match.group(0))
            if len(candidate) < 8:
                continue
            if candidate not in matches:
                matches.append(candidate)
            if len(matches) >= limit:
                return matches

    return matches
