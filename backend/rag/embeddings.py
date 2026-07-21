from sentence_transformers import SentenceTransformer


_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def _get_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(
            _MODEL_NAME
        )

    return _model


def warm_embedding_model():
    _get_model()


def _prepare_text(chunk):
    """
    Embed child chunks only.

    Example:

    [METHODOLOGY]

    Transformer encoder...
    """

    if isinstance(chunk, str):
        return chunk

    return (
        f"[{chunk['section'].upper()}]\n\n"
        f"{chunk['child_text']}"
    )


def create_embeddings(chunks):

    model = _get_model()

    texts = [
        _prepare_text(chunk)
        for chunk in chunks
    ]

    return model.encode(
        texts,
        convert_to_tensor=False
    )
