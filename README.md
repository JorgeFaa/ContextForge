# ContextForge

Pipeline de RAG (Retrieval-Augmented Generation) con **búsqueda híbrida**: combina similitud vectorial (embeddings) y búsqueda por palabras clave (full-text search) para recuperar contexto relevante de documentos y responder preguntas con un LLM, citando las fuentes usadas.

Construido como proyecto de aprendizaje/portafolio para explorar de punta a punta el diseño de un sistema RAG: ingestión de documentos, chunking, embeddings, indexación en base de datos vectorial, fusión de rankings, y generación aumentada por contexto — con proveedor de LLM intercambiable (local con Ollama, o vía API con Anthropic).

## Alcance del proyecto

**Incluye:**
- Ingesta de documentos `.pdf`, `.txt` y `.md`
- Chunking con overlap configurable
- Embeddings semánticos locales (`sentence-transformers`)
- Almacenamiento en PostgreSQL con `pgvector`
- Búsqueda híbrida: distancia coseno (HNSW) + full-text search nativo de Postgres (GIN + `tsvector`), combinadas con Reciprocal Rank Fusion (RRF)
- Generación de respuestas con citación de fuentes, con proveedor de LLM configurable (Ollama local o Anthropic)
- API REST (FastAPI) con dos endpoints: subir documentos y hacer preguntas

**No incluye (por ahora / fuera de alcance):**
- Interfaz gráfica (se usa la API directamente o la UI interactiva de FastAPI en `/docs`)
- Gestión de documentos (listar/eliminar) — solo ingesta y consulta
- Autenticación / multiusuario
- Chunking semántico (se usa tamaño fijo de palabras, no por secciones/párrafos)
- Re-ranking con modelos cross-encoder tras la fusión híbrida

## Arquitectura

```
Documento (.pdf/.txt/.md)
        │
        ▼
  [ingestion/loader.py]      → extrae texto plano
        │
        ▼
  [ingestion/chunker.py]     → parte en fragmentos con overlap
        │
        ▼
  [retrieval/embeddings.py]  → genera vectores (sentence-transformers)
        │
        ▼
  [db/ (Postgres + pgvector)]  ← se guarda texto + vector + índice de texto completo

Pregunta del usuario
        │
        ▼
  [retrieval/hybrid.py] ──┬── búsqueda vectorial (pgvector, distancia coseno)
                          └── búsqueda por palabras clave (Postgres FTS + BM25-like ts_rank)
                          │
                          ▼
                  fusión con Reciprocal Rank Fusion (RRF)
                          │
                          ▼
  [generation/answer.py]  → arma el prompt con los chunks recuperados
                          → llama al LLM (Ollama u Anthropic)
                          │
                          ▼
                  respuesta + fuentes citadas
```

## Stack técnico

| Componente | Tecnología | Por qué |
|---|---|---|
| Backend / API | FastAPI | Validación automática con Pydantic, documentación interactiva gratis (`/docs`) |
| Base de datos | PostgreSQL + `pgvector` | Un solo motor para vectores y texto completo, sin sumar infraestructura extra |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Corre localmente en CPU, rápido, 384 dimensiones |
| Búsqueda vectorial | Índice HNSW (`pgvector`) | Búsqueda aproximada de vecinos más cercanos, escalable |
| Búsqueda por palabras clave | Full-text search nativo de Postgres (`tsvector` + GIN) | Evita cargar todo en memoria; ya indexado, ya en la misma base de datos |
| Fusión de resultados | Reciprocal Rank Fusion (RRF) | Combina ambos rankings sin necesitar normalizar puntajes de escalas distintas |
| LLM | Ollama (local, `llama3.1`) o Anthropic (API) | Intercambiable vía `.env`, para desarrollar sin depender de una API key |

## Instalación

Requiere: Python 3.12, Docker, y opcionalmente Ollama (para correr el LLM localmente sin API key).

```bash
git clone <este-repo>
cd ContextForge

python3.12 -m venv venv
source venv/bin/activate
pip install -e .

cp .env.example .env
# edita .env: si usas Ollama, no necesitas API key.
# si usas Anthropic, agrega ANTHROPIC_API_KEY y cambia LLM_PROVIDER=anthropic

docker compose up -d          # levanta Postgres + pgvector
python -c "from app.db.session import init_db; init_db()"

# si usas Ollama:
ollama pull llama3.1

uvicorn app.main:app --reload
```

## Uso

Interfaz interactiva en `http://localhost:8000/docs`, o vía `curl`:

**Subir un documento:**
```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@ruta/al/documento.pdf"
```

**Hacer una pregunta:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿De qué trata el documento?", "top_k": 3}'
```

Respuesta:
```json
{
  "answer": "...",
  "sources": ["chunk 12 (doc 1)", "chunk 45 (doc 1)"]
}
```

## Evaluación

Se probó el pipeline con el PDF de la trilogía *Nacidos de la Bruma* (Brandon Sanderson, ~1.5M de caracteres) usando `llama3.1:8b` local, variando `top_k` entre 1, 3 y 5, sobre 10 preguntas de distinto tipo (factuales, terminología específica, relacionales, y una pregunta trampa fuera del corpus). Registro completo en [`pruebas/prueba.txt`](pruebas/prueba.txt).

**Hallazgos principales:**
- `top_k=3` fue el punto de mejor balance entre completitud y precisión en la mayoría de los casos; `top_k=1` tiende a quedarse corto, y `top_k=5` incrementó la tasa de alucinaciones.
- Preguntas sobre terminología específica del libro (ej. "¿Qué es el Atium?", "¿Qué es la Hemalurgia?") obtuvieron respuestas más precisas — consistente con que el componente de búsqueda por palabras clave del sistema híbrido es más efectivo con vocabulario exacto y poco común.
- Preguntas que requieren síntesis narrativa a través de toda la trilogía (ej. evolución de una relación entre personajes a lo largo de tres libros) tuvieron peor desempeño — limitación esperada de RAG por fragmentos, que recupera chunks puntuales y no reconstruye arcos narrativos completos.
- Patrón de alucinación identificado: el modelo, al no encontrar el título exacto de un libro en el contexto recuperado, lo completaba con datos de su conocimiento previo (inventando títulos incorrectos) en vez de omitir esa información — señal de que el prompt de sistema puede reforzarse aún más.
- La pregunta trampa (sobre un cuarto libro inexistente en la trilogía) fue correctamente identificada como fuera de contexto en todos los `top_k` probados.

**Mejora en curso:** configurar explícitamente la ventana de contexto de Ollama (`num_ctx`), dado que el valor por defecto puede truncar silenciosamente el contexto cuando se recuperan más chunks, lo cual podría explicar el aumento de alucinaciones observado en `top_k=5`.

## Estructura del proyecto

```
app/
├── main.py                  # endpoints de FastAPI
├── config.py                 # configuración vía variables de entorno
├── ingestion/
│   ├── loader.py              # extracción de texto (PDF/TXT/MD)
│   └── chunker.py             # partición en fragmentos con overlap
├── retrieval/
│   ├── embeddings.py          # generación de embeddings
│   ├── vector_store.py        # inserción y búsqueda vectorial (pgvector)
│   ├── keyword_store.py       # búsqueda por palabras clave (Postgres FTS)
│   └── hybrid.py              # fusión de resultados (RRF)
├── generation/
│   └── answer.py               # construcción de prompt y llamada al LLM
└── db/
    ├── schema.sql              # definición de tablas e índices
    ├── session.py               # conexión y pool a Postgres
    └── models.py                 # representación en Python de las filas
```
