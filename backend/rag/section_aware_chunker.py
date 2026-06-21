import re
from typing import Dict, List, Any, Union
from backend.rag.models import ChunkMetadata, ChunkedSection


# Regex patterns to detect research paper sections
SECTION_PATTERNS = {
    "abstract": r"^\s*abstract\s*$",
    "introduction": r"^\s*(introduction|intro)\s*$",
    "related_work": r"^\s*(related\s+work|related\s+works|background)\s*$",
    "background": r"^\s*background\s*$",
    "methodology": r"^\s*(methodology|method|methods|approach)\s*$",
    "experimental_setup": r"^\s*(experimental\s+setup|experiment\s+setup|setup)\s*$",
    "experiments": r"^\s*experiments?\s*$",
    "results": r"^\s*results?\s*$",
    "discussion": r"^\s*discussion\s*$",
    "limitations": r"^\s*limitations?\s*$",
    "future_work": r"^\s*(future\s+work|future\s+works|future\s+directions?)\s*$",
    "conclusion": r"^\s*(conclusion|conclusions?)\s*$",
}


def detect_sections(text: str) -> Dict[str, Dict[str, Union[int, str]]]:
    """
    Detect sections in text and return their boundaries and content.

    Args:
        text: Full document text

    Returns:
        Dictionary with section names as keys and section info (start, end, content, order)
    """
    lines = text.split('\n')
    sections = {}
    section_order = 0
    section_starts = []

    # Find all section headers
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        # Check if this line matches any section pattern
        for section_name, pattern in SECTION_PATTERNS.items():
            if re.match(pattern, stripped_line, re.IGNORECASE):
                section_starts.append((i, section_name, section_order))
                section_order += 1
                break

    # If no sections found, return entire text as "full_text"
    if not section_starts:
        return {
            "full_text": {
                "start": 0,
                "end": len(text),
                "content": text,
                "order": 0
            }
        }

    # Extract section boundaries and content
    for idx, (line_idx, section_name, order) in enumerate(section_starts):
        # Calculate start position (character position at start of section header)
        start_pos = sum(len(lines[i]) + 1 for i in range(line_idx))

        # Calculate end position (character position at start of next section or end of text)
        if idx + 1 < len(section_starts):
            next_line_idx = section_starts[idx + 1][0]
            end_pos = sum(len(lines[i]) + 1 for i in range(next_line_idx))
        else:
            end_pos = len(text)

        # Extract content (everything after the header until next section)
        if idx + 1 < len(section_starts):
            next_line_idx = section_starts[idx + 1][0]
            content_lines = lines[line_idx + 1:next_line_idx]
        else:
            content_lines = lines[line_idx + 1:]

        content = '\n'.join(content_lines).strip()

        sections[section_name] = {
            "start": start_pos,
            "end": end_pos,
            "content": content,
            "order": order
        }

    return sections


def chunk_section_text(section_text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """
    Chunk a section text using word-based windowing.

    Args:
        section_text: The text of a single section
        chunk_size: Target number of words per chunk
        overlap: Number of words to overlap between chunks

    Returns:
        List of chunk strings
    """
    # Validate parameters to prevent infinite loop
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    # Split into words
    words = section_text.split()

    if len(words) <= chunk_size:
        return [section_text]

    chunks = []
    step = chunk_size - overlap

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words))

        # If we've reached the end, break
        if end == len(words):
            break

        start += step

    return chunks


def create_section_aware_chunks(text: str, chunk_size: int = 600, overlap: int = 100) -> List[ChunkedSection]:
    """
    Main entry point: detect sections and chunk each independently.

    Args:
        text: Full document text
        chunk_size: Target number of words per chunk
        overlap: Number of words to overlap between chunks

    Returns:
        List of ChunkedSection objects with ChunkMetadata
    """
    # Detect sections
    sections = detect_sections(text)

    # Sort sections by order
    sorted_sections = sorted(sections.items(), key=lambda x: x[1]["order"])

    chunked_sections = []

    for section_name, section_info in sorted_sections:
        section_content = section_info["content"]
        section_index = section_info["order"]

        # Chunk this section
        chunks = chunk_section_text(section_content, chunk_size=chunk_size, overlap=overlap)

        # Create ChunkMetadata objects
        chunk_metadata_list = []
        for chunk_idx, chunk_text in enumerate(chunks):
            metadata = ChunkMetadata(
                text=chunk_text,
                section=section_name,
                section_index=section_index,
                chunk_index=chunk_idx
            )
            chunk_metadata_list.append(metadata)

        # Create ChunkedSection
        chunked_section = ChunkedSection(
            section_name=section_name,
            section_index=section_index,
            chunks=chunk_metadata_list
        )
        chunked_sections.append(chunked_section)

    return chunked_sections
