from sentence_transformers import SentenceTransformer


_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _get_model():
    """Lazy-load and cache the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def create_embeddings(chunks):
    """Convert text chunks into embeddings."""
    model = _get_model()
    return model.encode(chunks, convert_to_tensor=False)
