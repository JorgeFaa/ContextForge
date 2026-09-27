# app/main.py
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from app.generation.answer import generate_answer, list_available_models
from app.ingestion.chunker import chunk_text
from app.ingestion.loader import load_document
from app.retrieval.embeddings import embed_batch
from app.retrieval.hybrid import hybrid_search
from app.retrieval.vector_store import insert_chunks, insert_document, delete_document, list_documents

app = FastAPI(title="ContextForge")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    model: str | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

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


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    chunks = hybrid_search(request.question, top_k = request.top_k)
    try:
        answer = generate_answer(request.question, chunks, model = request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    sources = [f"chunk {chunk.id} (doc {chunk.document_id})" for chunk in chunks]
    return QueryResponse(answer=answer, sources=sources)

@app.get("/documents")
def list_documents_endpoint() -> list[dict]:
    return list_documents()


@app.delete("/documents/{document_id}")
def delete_document_endpoint(document_id: int) -> dict[str, bool]:
    deleted = delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    return {"deleted": deleted}


@app.get("/models")
def get_models() -> list[str]:
    return list_available_models()