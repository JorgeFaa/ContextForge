# app/main.py
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from app.generation.answer import generate_answer
from app.ingestion.chunker import chunk_text
from app.ingestion.loader import load_document
from app.retrieval.embeddings import embed_batch
from app.retrieval.hybrid import hybrid_search
from app.retrieval.vector_store import insert_chunks, insert_document

app = FastAPI(title="ContextForge")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(file: UploadFile) -> dict[str, int | str]:
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    text = load_document(tmp_path)
    chunks = chunk_text(text)
    embeddings = embed_batch(chunks)

    document_id = insert_document(file.filename)
    insert_chunks(document_id, chunks, embeddings)

    tmp_path.unlink()

    return {"document_id": document_id, "chunks_indexed": len(chunks)}

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    chunks = hybrid_search(request.question, top_k = request.top_k)
    answer = generate_answer(request.question, chunks)
    sources = [f"chunk {chunk.id} (doc {chunk.document_id})" for chunk in chunks]
    return QueryResponse(answer=answer, sources=sources)