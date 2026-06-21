from pydantic import BaseModel
from typing import List


class ChunkMetadata(BaseModel):
    """Metadata for a single chunk of text."""
    text: str
    section: str  # e.g., "Abstract", "Introduction", "Methods"
    section_index: int  # 0 = Abstract, 1 = Introduction, etc.
    chunk_index: int  # Position within the section (0-indexed)


class ChunkedSection(BaseModel):
    """Collection of chunks from a single paper section."""
    section_name: str
    section_index: int
    chunks: List[ChunkMetadata]
