import pytest
from backend.rag.models import ChunkMetadata, ChunkedSection

def test_chunk_metadata_structure():
    """Verify ChunkMetadata has required fields."""
    chunk = ChunkMetadata(
        text="This is a test chunk.",
        section="Introduction",
        section_index=1,
        chunk_index=0
    )
    assert chunk.text == "This is a test chunk."
    assert chunk.section == "Introduction"
    assert chunk.section_index == 1
    assert chunk.chunk_index == 0

def test_chunked_section_structure():
    """Verify ChunkedSection holds all chunks from one section."""
    chunks = [
        ChunkMetadata(text="First chunk.", section="Methods", section_index=2, chunk_index=0),
        ChunkMetadata(text="Second chunk.", section="Methods", section_index=2, chunk_index=1),
    ]
    section = ChunkedSection(section_name="Methods", section_index=2, chunks=chunks)
    assert section.section_name == "Methods"
    assert len(section.chunks) == 2
    assert section.chunks[0].text == "First chunk."
