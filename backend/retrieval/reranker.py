from sentence_transformers import CrossEncoder


class Reranker:
    _model = None

    def __init__(self):
        self.model = self._get_model()

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            cls._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return cls._model

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


def warm_reranker_model():
    Reranker._get_model()
