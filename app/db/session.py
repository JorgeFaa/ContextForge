# app/db/session.py
from contextlib import contextmanager
from pathlib import Path

import psycopg2
from psycopg2 import pool
from pgvector.psycopg2 import register_vector

from app.config import settings

_pool = pool.SimpleConnectionPool(1, 10, dsn=settings.database_url)

@contextmanager
def get_connection():
    conn = _pool.getconn()
    register_vector(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


def init_db() -> None:
    """Crea las tablas e indices si no existen"""
    schema_path = Path(__file__).parent / "schema.sql"
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(schema_path.read_text())