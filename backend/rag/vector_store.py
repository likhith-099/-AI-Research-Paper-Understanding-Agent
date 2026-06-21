from typing import List, Dict, Union

import faiss
import numpy as np


class VectorStore:
    """
    Vector store using FAISS for efficient similarity search.

    Supports both plain string chunks (legacy) and dicts with metadata.
    """

    def __init__(self, embeddings: List[np.ndarray], chunks: List[Union[str, Dict]]):
        """
        Initialize VectorStore with embeddings and chunks.

        Args:
            embeddings: List of embedding vectors (as numpy arrays or lists).
            chunks: List of chunks in one of two formats:
                - Plain strings (legacy): ["chunk1 text", "chunk2 text", ...]
                - Dicts with metadata: [
                    {"text": "...", "section": "Abstract", ...},
                    {"text": "...", "section": "Intro", ...},
                    ...
                  ]

        The VectorStore stores chunks as-is and returns the same format on search.
        This enables both backward compatibility (string chunks) and metadata support.
        """
        self.chunks = chunks

        dimension = len(embeddings[0])

        self.index = faiss.IndexFlatL2(dimension)

        self.embeddings = np.array(embeddings).astype("float32")

        self.index.add(self.embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Union[str, Dict]]:
        """
        Returns most relevant chunks for a query.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of top results to return.

        Returns:
            List of chunks in the same format as input:
            - List of strings if chunks were strings
            - List of dicts if chunks had metadata
        """

        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for i in indices[0]:
            results.append(self.chunks[i])

        return results