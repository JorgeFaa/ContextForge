# app/retrieval/embeddings.py

from sentence_transformers import SentenceTransformer

from app.config import settings

_model = SentenceTransformer(settings.embedding_model)


def embed_text(text: str) -> list[float]:
    """Convierte un texto en su vector de embedding"""
    return _model.encode(text, normalize_embeddings=True).tolist()

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Convierte varios textos en sus vectores de embedding, en un solo lote"""
    return _model.encode(texts, normalize_embeddings=True).tolist()