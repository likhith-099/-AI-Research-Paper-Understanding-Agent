from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, chunks):

        self.chunks = chunks

        if not chunks:
            self.bm25 = None
            return

        corpus = []

        for chunk in chunks:

            if isinstance(chunk, dict):

                corpus.append(
                    chunk["child_text"].split()
                )

            else:

                corpus.append(
                    chunk.split()
                )

        self.bm25 = BM25Okapi(corpus)

    def search(
        self,
        query,
        top_k=50,
        allowed_sections=None,
        equation_only: bool = False
    ):
        if self.bm25 is None:
            return []

        query_tokens = query.split()

        scores = self.bm25.get_scores(
            query_tokens
        )

        normalized_sections = None
        if allowed_sections:
            normalized_sections = {
                section.lower()
                for section in allowed_sections
            }

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        filtered_ranked = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            if normalized_sections and str(chunk.get("section", "")).lower() not in normalized_sections:
                continue
            if equation_only and not chunk.get("has_equation"):
                continue
            filtered_ranked.append((idx, score))

        for idx, score in filtered_ranked[:top_k]:
            chunk = self.chunks[idx]

            results.append(
                (
                    chunk,
                    float(score)
                )
            )

        return results
