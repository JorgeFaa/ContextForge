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

def list_documents() -> list[dict]:
    """Devuelve todos los documentos con el número de chunks indexados"""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.id, d.filename, d.created_at, COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.created_at DESC
            """
        )
        rows = cur.fetchall()
        return [
            {"id": r[0], "filename": r[1], "created_at": r[2].isoformat(), "chunk_count": r[3]}
            for r in rows
        ]

def delete_document(document_id: int) -> bool:
    """Elimina un documento y sus chunks asociados (cascada). Devuelve True si existia."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
        return cur.rowcount > 0