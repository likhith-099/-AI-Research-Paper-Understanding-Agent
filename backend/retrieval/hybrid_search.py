class HybridSearch:

    def fuse_scores(
        self,
        dense_results,
        sparse_results
    ):

        scores = {}

        for rank, chunk in enumerate(dense_results):

            child_id = chunk["child_id"]

            scores.setdefault(
                child_id,
                {
                    "chunk": chunk,
                    "score": 0.0
                }
            )

            scores[child_id]["score"] += (
                1.0 / (rank + 1)
            )

        for rank, (chunk, _) in enumerate(
            sparse_results
        ):

            child_id = chunk["child_id"]

            scores.setdefault(
                child_id,
                {
                    "chunk": chunk,
                    "score": 0.0
                }
            )

            scores[child_id]["score"] += (
                1.0 / (rank + 1)
            )

        ranked = sorted(
            scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return [
            item["chunk"]
            for item in ranked
        ]