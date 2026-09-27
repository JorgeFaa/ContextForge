# app/retrieval/vector_store.py
from app.db.session import get_connection
from app.db.models import Chunk


def insert_document(filename: str) -> int:
    """Inserta un documento y devuelve su id"""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (filename) VALUES (%s) RETURNING id",
            (filename,),
        )
        return cur.fetchone()[0]


def insert_chunks(document_id: int, texts: list[str], embeddings: list[list[float]]) -> None:
    """Inserta varios chunks de un documento junto con sus embeddings"""
    with get_connection() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (document_id, text, embedding) VALUES (%s, %s, %s)",
            list(zip([document_id] * len(texts), texts, embeddings))
        )


def search_by_vector(query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
    """Devuelve los 'top_k' chunks más cercanos al embedding de consulta"""

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, document_id, text, embedding
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, top_k),
        )
        rows = cur.fetchall()
        return [Chunk(id = r[0], document_id = r[1], text = r[2], embedding = r[3].to_list()) for r in rows]