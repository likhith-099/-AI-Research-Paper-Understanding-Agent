from typing import List, Dict, Union, Optional, Sequence

import faiss
import numpy as np


class VectorStore:
    """
    FAISS Vector Store

    Stores:
    - Parent/Child chunks
    - Metadata
    - Embeddings
    """

    def __init__(
        self,
        embeddings: List[np.ndarray],
        chunks: List[Union[str, Dict]]
    ):
        if embeddings is None or chunks is None or len(embeddings) == 0 or len(chunks) == 0:
            raise ValueError("VectorStore requires at least one embedding and one chunk.")

        self.chunks = chunks

        dimension = len(embeddings[0])

        self.index = faiss.IndexFlatL2(dimension)

        self.embeddings = np.array(
            embeddings
        ).astype("float32")

        self.index.add(
            self.embeddings
        )

    def _candidate_indices(
        self,
        allowed_sections: Optional[Sequence[str]] = None,
        equation_only: bool = False,
    ):
        normalized_sections = None
        if allowed_sections:
            normalized_sections = {
                section.lower()
                for section in allowed_sections
            }

        indices = []
        for idx, chunk in enumerate(self.chunks):
            if normalized_sections and str(chunk.get("section", "")).lower() not in normalized_sections:
                continue
            if equation_only and not chunk.get("has_equation"):
                continue
            indices.append(idx)
        return indices

    def _search_subset(self, query_embedding, candidate_indices, top_k):
        if not candidate_indices:
            return []

        subset_embeddings = self.embeddings[candidate_indices]
        dimension = subset_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(subset_embeddings.astype("float32"))

        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = index.search(query_embedding, min(top_k, len(candidate_indices)))

        results = []
        for local_idx, distance in zip(indices[0], distances[0]):
            if local_idx == -1:
                continue
            original_idx = candidate_indices[local_idx]
            results.append(
                (
                    self.chunks[original_idx],
                    float(distance),
                    original_idx,
                )
            )
        return results

    def search(
        self,
        query_embedding,
        top_k=50,
        allowed_sections=None,
        equation_only: bool = False
    ):
        results_with_scores = self.search_with_scores(
            query_embedding,
            top_k=top_k,
            allowed_sections=allowed_sections,
            equation_only=equation_only,
        )
        return [
            chunk
            for chunk, _, _ in results_with_scores
        ]

    def search_with_scores(
        self,
        query_embedding,
        top_k=50,
        allowed_sections=None,
        equation_only: bool = False
    ):
        candidate_indices = self._candidate_indices(
            allowed_sections=allowed_sections,
            equation_only=equation_only,
        )
        return self._search_subset(
            query_embedding,
            candidate_indices,
            top_k,
        )
