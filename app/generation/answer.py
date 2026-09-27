# app/generation/answer.py
import httpx
from anthropic import Anthropic

from app.config import settings
from app.db.models import Chunk

SYSTEM_PROMPT = (
    "Responde la pregunta del usuario basándote únicamente en el contexto proporcionado. "
    "Si la respuesta no está en el contexto, dilo explícitamente en vez de inventar una respuesta."
)

def build_prompt(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(f"[Fragmento {i+1}]\n{chunk.text}" for i, chunk in enumerate(chunks))
    return f"Contexto:\n{context}\n\nPregunta: {question}"

def generate_answer(question:str, chunks: list[Chunk]) -> str:
    prompt = build_prompt(question, chunks)

    if settings.llm_provider == "ollama":
        return _call_ollama(prompt)
    return _call_anthropic(prompt)

def _call_ollama(prompt: str) -> str:
    response = httpx.post(
        f"{settings.ollama_base_url}/api/generate",
        json={
            "model": settings.ollama_model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": 8192},
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"]

def _call_anthropic(prompt: str) -> str:
    client = Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text