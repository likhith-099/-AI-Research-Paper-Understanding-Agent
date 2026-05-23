import faiss
import numpy as np


class VectorStore:

    def __init__(self, embeddings, chunks):
        self.chunks = chunks

        dimension = len(embeddings[0])

        self.index = faiss.IndexFlatL2(dimension)

        self.embeddings = np.array(embeddings).astype("float32")

        self.index.add(self.embeddings)

    def search(self, query_embedding, top_k=5):
        """
        Returns most relevant chunks for a query
        """

        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for i in indices[0]:
            results.append(self.chunks[i])

        return results