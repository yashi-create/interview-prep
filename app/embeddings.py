import os
from functools import lru_cache
from fastembed import TextEmbedding

# Same model, same 384-dim output as before - just run through onnxruntime
# instead of torch. sentence-transformers pulls in full PyTorch, whose CPU
# runtime alone reserves 300-500MB RSS; combined with the rest of the app
# that blew past Render's 512MB free-tier cap. fastembed has no torch
# dependency, so the same embeddings fit comfortably in that budget.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _get_model() -> TextEmbedding:
    """Load the model once and cache it - loading is the slow part, so we
    don't want to repeat it on every request."""
    return TextEmbedding(model_name=EMBEDDING_MODEL)


def embed_text(text: str) -> list[float]:
    """Turn a string into a 384-dim embedding vector (as a plain list, so
    it's directly storable in the pgvector column). pgvector's cosine
    distance operator normalizes internally, so no explicit normalization
    is needed here."""
    model = _get_model()
    vector = next(model.embed([text]))
    return vector.tolist()
