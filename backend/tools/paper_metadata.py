from __future__ import annotations

import re
from typing import Dict, List

import fitz  # PyMuPDF


YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def _score_page_overlap(chunk_tokens: List[str], page_tokens: List[str]) -> float:
    if not chunk_tokens or not page_tokens:
        return 0.0

    chunk_set = set(chunk_tokens)
    page_set = set(page_tokens)
    overlap = len(chunk_set & page_set)
    return overlap / max(1, len(chunk_set))


def _clean_line(line: str) -> str:
    return _normalize_whitespace(line).strip(" .,:;")


def _infer_title(first_page_text: str, metadata: Dict[str, str]) -> str:
    title = _clean_line(str(metadata.get("title") or ""))
    if title and title.lower() not in {"untitled", "unknown"}:
        return title

    lines = [
        _clean_line(line)
        for line in first_page_text.splitlines()
    ]
    candidates = [
        line
        for line in lines[:20]
        if line
        and len(line.split()) >= 3
        and len(line) <= 180
        and "abstract" not in line.lower()
        and "arxiv" not in line.lower()
        and not re.search(r"\b\d+\b", line)
    ]
    if candidates:
        return max(candidates, key=len)

    if lines:
        return lines[0]

    return "Unknown title"


def _infer_authors(first_page_text: str, metadata: Dict[str, str]) -> str:
    authors = _clean_line(str(metadata.get("author") or ""))
    if authors and authors.lower() not in {"anonymous", "unknown"}:
        return authors

    lines = [
        _clean_line(line)
        for line in first_page_text.splitlines()
    ]
    for index, line in enumerate(lines[:25]):
        lowered = line.lower()
        if not line or "abstract" in lowered:
            continue
        if re.search(r"\b(and|,)\b", lowered) and len(line.split()) <= 18:
            return line
        if index > 0 and len(line.split()) <= 14 and any(char.isalpha() for char in line):
            return line

    return "Unknown authors"


def _infer_year(first_page_text: str, metadata: Dict[str, str]) -> str:
    for key in ("creationDate", "modDate", "subject"):
        value = str(metadata.get(key) or "")
        match = YEAR_PATTERN.search(value)
        if match:
            return match.group(0)

    match = YEAR_PATTERN.search(first_page_text or "")
    if match:
        return match.group(0)

    return "Unknown year"


def extract_pdf_metadata(pdf_path: str) -> Dict[str, object]:
    doc = fitz.open(pdf_path)
    try:
        pdf_metadata = doc.metadata or {}
        page_count = int(doc.page_count)
        first_page_text = doc[0].get_text("text") if page_count else ""

        return {
            "title": _infer_title(first_page_text, pdf_metadata),
            "authors": _infer_authors(first_page_text, pdf_metadata),
            "year": _infer_year(first_page_text, pdf_metadata),
            "pages": page_count,
        }
    finally:
        doc.close()


def extract_pdf_pages(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    try:
        return [_normalize_whitespace(page.get_text("text")) for page in doc]
    finally:
        doc.close()


def annotate_chunk_pages(chunks: List[Dict], page_texts: List[str]) -> List[Dict]:
    page_tokens = [_tokenize(page_text) for page_text in page_texts]

    for chunk in chunks:
        text = str(chunk.get("child_text") or "")
        tokens = _tokenize(text)
        if not tokens:
            chunk["pages"] = []
            continue

        scores = []
        chunk_prefix = _normalize_whitespace(text[:240]).lower()
        for page_number, page_text in enumerate(page_texts, start=1):
            page_text_lower = page_text.lower()
            if chunk_prefix and chunk_prefix[:60] in page_text_lower:
                scores.append((page_number, 1.0))
                continue
            score = _score_page_overlap(tokens, page_tokens[page_number - 1])
            if score > 0.08:
                scores.append((page_number, score))

        if not scores:
            best_page = 1
            best_score = 0.0
            for page_number, page_token_list in enumerate(page_tokens, start=1):
                score = _score_page_overlap(tokens, page_token_list)
                if score > best_score:
                    best_page = page_number
                    best_score = score
            scores = [(best_page, best_score)] if best_page else []

        selected_pages = sorted({page_number for page_number, _ in scores[:2]})
        chunk["pages"] = selected_pages

    return chunks

