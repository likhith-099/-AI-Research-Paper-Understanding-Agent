from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(
        self,
        query,
        chunks,
        top_k=20
    ):
        ranked = self.rerank_with_scores(query, chunks, top_k=top_k)
        return [
            item["chunk"]
            for item in ranked
        ]

    def rerank_with_scores(
        self,
        query,
        chunks,
        top_k=20
    ):

        if not chunks:
            return []

        pairs = []

        for chunk in chunks:

            pairs.append(
                [
                    query,
                    chunk["child_text"]
                ]
            )

        scores = self.model.predict(
            pairs
        )

        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {
                "chunk": chunk,
                "score": float(score),
            }
            for chunk, score in ranked[:top_k]
        ]
