#app/ingestion/chunker.py


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Parte del texto en fragmentos de 'chunk_size' palabras, con 'overlap' palabras compartidas entre fragmentos consecutivos"""
    words = text.split()
    if not words:
        return []


    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks