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
- Generación de respuestas con citación de fuentes, con proveedor de LLM configurable (Ollama local o Anthropic) y selección de modelo por request
- Gestor de documentos: subir, listar y eliminar (con borrado en cascada de sus chunks)
- API REST completa (FastAPI) — ver [Endpoints](#endpoints)
- Suite de tests unitarios (`pytest`) para la lógica pura del pipeline

**No incluye (por ahora / fuera de alcance):**
- Interfaz gráfica — este repositorio es **solo la API**. La interfaz visual vive en un repositorio aparte, **ContextForgeStudio**, que consume esta API (mismo patrón que DataForge/DataForgeStudio)
- Autenticación / multiusuario
- Chunking semántico (se usa tamaño fijo de palabras, no por secciones/párrafos)
- Re-ranking con modelos cross-encoder tras la fusión híbrida
- Tests de integración contra una base de datos real (la capa de acceso a datos se valida hoy con pruebas manuales, ver [Evaluación](#evaluación))

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
git clone https://github.com/JorgeFaa/ContextForge
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

## Endpoints

Documentación interactiva completa en `http://localhost:8000/docs`.

| Método | Ruta | Qué hace |
|---|---|---|
| `GET` | `/health` | Confirma que el servicio está corriendo |
| `POST` | `/documents` | Sube un documento (`.pdf`/`.txt`/`.md`), lo indexa (chunking + embeddings) |
| `GET` | `/documents` | Lista los documentos indexados, con su número de chunks |
| `DELETE` | `/documents/{id}` | Elimina un documento y sus chunks (borrado en cascada) |
| `GET` | `/models` | Lista los modelos de Ollama disponibles localmente |
| `POST` | `/query` | Busca contexto híbrido y genera una respuesta citando fuentes |

## Uso

**Subir un documento:**
```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@ruta/al/documento.pdf"
```

**Hacer una pregunta** (el campo `model` es opcional; si se omite, usa el definido en `.env`):
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿De qué trata el documento?", "top_k": 3, "model": "llama3.1:latest"}'
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

**Mejora aplicada:** se configuró explícitamente la ventana de contexto de Ollama (`num_ctx`), ya que el valor por defecto truncaba silenciosamente el contexto al recuperar más chunks — esto reducía la información real disponible para el modelo sin previo aviso. Tras el cambio, se observó una disminución en las alucinaciones al repetir las mismas preguntas con `top_k=5`.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

La suite cubre la lógica pura y determinista del pipeline — código sin dependencias externas (base de datos, red, modelos de ML), donde un test puede dar el mismo resultado siempre con la misma entrada:

- `tests/test_chunker.py` — partición de texto en fragmentos y comportamiento del overlap
- `tests/test_hybrid.py` — fusión de resultados con Reciprocal Rank Fusion (que un chunk presente en ambas búsquedas gane frente a uno presente en solo una, deduplicación, respeto de `top_k`)
- `tests/test_loader.py` — extracción de texto por tipo de archivo y manejo de extensiones no soportadas

Deliberadamente fuera de esta suite (dependen de infraestructura externa — Postgres, el modelo de embeddings, Ollama/Anthropic): `vector_store.py`, `keyword_store.py`, `embeddings.py`, `answer.py`. Esa capa se valida con las pruebas manuales documentadas en la sección de Evaluación.

## Proyectos relacionados

- **ContextForgeStudio** *(repositorio aparte, en planeación)* — interfaz visual que consume esta API. Mismo patrón de separación API/interfaz usado en DataForge / DataForgeStudio.

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
