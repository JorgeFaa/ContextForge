# app/retrieval/keyword_store.py
from app.db.session import get_connection
from app.db.models import Chunk


def search_by_keyword(query: str, top_k: int = 5) -> list[Chunk]:
    """Devuelve los 'top_k' chunks con mejor coincidencia de palabras clave"""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, document_id, text, embedding,
                ts_rank(text_search, plainto_tsquery('spanish', %s)) AS rank
            FROM chunks
            WHERE text_search @@ plainto_tsquery('spanish', %s)
            ORDER BY rank DESC
            LIMIT %s
            """,
            (query, query, top_k),
        )
        rows = cur.fetchall()
        return [Chunk(id = r[0], document_id = r[1], text = r[2], embedding = r[3].to_list()) for r in rows]