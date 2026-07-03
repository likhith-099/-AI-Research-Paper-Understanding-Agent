from typing import Dict, List

from rag.models import ChunkMetadata, ChunkedSection
from tools.section_detector import detect_sections_verbose


def detect_sections(text: str) -> Dict[str, Dict[str, object]]:
    verbose = detect_sections_verbose(text)
    return verbose["sections"]


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
        canonical_section = section_info.get("canonical_section", section_name)

        # Chunk this section
        chunks = chunk_section_text(section_content, chunk_size=chunk_size, overlap=overlap)

        # Create ChunkMetadata objects
        chunk_metadata_list = []
        for chunk_idx, chunk_text in enumerate(chunks):
            metadata = ChunkMetadata(
                text=chunk_text,
                section=canonical_section,
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
