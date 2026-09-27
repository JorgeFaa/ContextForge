# app/retrieval/hybrid.py
from app.db.models import Chunk
from app.retrieval.embeddings import embed_text
from app.retrieval.keyword_store import search_by_keyword
from app.retrieval.vector_store import search_by_vector

RRF_K = 60

def fuse_results(vector_results: list[Chunk], keyword_results: list[Chunk], top_k: int) -> list[Chunk]:
    """Combina dos listas de chunks ya ordenadas por relevancai, usando reciprocal rank fusion"""
    scores: dict[int, float] = {}
    chunks_by_id: dict[int, Chunk] = {}
    
    for rank, chunk in enumerate(vector_results):
        scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (RRF_K + rank)
        chunks_by_id[chunk.id] = chunk
    
    for rank, chunk in enumerate(keyword_results):
        scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (RRF_K + rank)
        chunks_by_id[chunk.id] = chunk

    ranked_ids = sorted(scores, key = scores.get, reverse = True)
    return [chunks_by_id[chunk_id] for chunk_id in ranked_ids[:top_k]]


def hybrid_search(query: str, top_k: int = 5, candidates_per_method: int = 20) -> list[Chunk]:
    """Busca `query` combinando resultados vectoriales y de palabras clave vía Reciprocal Rank Fusion."""
    query_embedding = embed_text(query)

    vector_results = search_by_vector(query_embedding, top_k = candidates_per_method)
    keyword_results = search_by_keyword(query, top_k = candidates_per_method)

    return fuse_results(vector_results, keyword_results, top_k)
