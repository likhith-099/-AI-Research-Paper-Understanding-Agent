import pytest
import numpy as np
from backend.rag.vector_store import VectorStore


def test_vector_store_stores_metadata():
    """Verify chunks with metadata are stored correctly."""
    chunks = [
        {"text": "First abstract chunk.", "section": "Abstract", "section_index": 0, "chunk_index": 0},
        {"text": "Second abstract chunk.", "section": "Abstract", "section_index": 0, "chunk_index": 1},
        {"text": "Introduction starts here.", "section": "Introduction", "section_index": 1, "chunk_index": 0},
    ]

    # Create dummy embeddings (384 dims like all-MiniLM-L6-v2)
    embeddings = [np.random.random(384).astype("float32") for _ in chunks]

    store = VectorStore(embeddings, chunks)

    assert len(store.chunks) == 3
    assert store.chunks[0]["section"] == "Abstract"
    assert store.chunks[1]["section"] == "Abstract"
    assert store.chunks[2]["section"] == "Introduction"


def test_vector_store_search_returns_metadata():
    """Verify search returns chunks with metadata intact."""
    chunks = [
        {"text": "First chunk about methods.", "section": "Methodology"},
        {"text": "Second chunk about methods.", "section": "Methodology"},
        {"text": "Results show improvement.", "section": "Results"},
    ]

    embeddings = [np.random.random(384).astype("float32") for _ in chunks]
    store = VectorStore(embeddings, chunks)

    query_embedding = np.random.random(384).astype("float32")
    results = store.search(query_embedding, top_k=2)

    assert len(results) == 2
    assert "section" in results[0]
    assert "text" in results[0]


def test_vector_store_legacy_string_chunks():
    """Verify backward compatibility with plain string chunks."""
    chunks = [
        "First chunk text",
        "Second chunk text",
        "Third chunk text",
    ]

    embeddings = [np.random.random(384).astype("float32") for _ in chunks]
    store = VectorStore(embeddings, chunks)

    assert len(store.chunks) == 3
    assert store.chunks[0] == "First chunk text"
    assert store.chunks[1] == "Second chunk text"

    query_embedding = np.random.random(384).astype("float32")
    results = store.search(query_embedding, top_k=2)

    assert len(results) == 2
    assert isinstance(results[0], str)
