from sentence_transformers import SentenceTransformer


class Retriever:

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve_context(self, query, top_k=5):
        """
        Search relevant chunks for a given query.
        """

        query_embedding = self.model.encode(query)

        results = self.vector_store.search(query_embedding, top_k)

        return "\n".join(results)