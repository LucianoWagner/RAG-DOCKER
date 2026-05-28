# AGENT.md — Contexto Completo del Proyecto RAG para Asistentes IA

> Este archivo fue diseñado para ser leído por un asistente de IA al inicio de una sesión.
> Contiene toda la información necesaria para entender el proyecto, sus decisiones de diseño,
> su estado actual y los próximos pasos, sin necesidad de leer el código completo.

---

## 1. RESUMEN EJECUTIVO

**Qué es:** Asistente conversacional de soporte técnico de Docker, basado en documentación oficial.
Usa RAG (Retrieval-Augmented Generation) con búsqueda híbrida, reranking, verificación de evidencia
y citas trazables. El corpus son los archivos `.md` de la documentación oficial de Docker.

**Qué NO es:** No es un chatbot genérico. Todos los prompts, metadata, evaluación y corpus están
diseñados exclusivamente para soporte técnico de Docker. Si el usuario pregunta algo fuera del corpus,
el sistema se abstiene (no alucina).

**Stack del proyecto:**
- Frontend: **OpenWebUI** (Docker, `:3000`)
- Vector Store: **ChromaDB** (Docker, `:8000`)
- LLM local/Embeddings: **Ollama** (Docker, `:11434`) — modelos: `llama3.1:8b`, `nomic-embed-text`
- Backend RAG: **FastAPI + LangChain** (Python local en `venv`, `:8080`)
- LLM en nube: **Groq API** — modelo activo: `llama-3.3-70b-versatile`
- Evaluación: **RAGAS** (entorno separado `venv-eval`, Python 3.11 requerido)

**Repo path:** `E:\ProyectoRagSoporteTecnico`

---

## 2. ARQUITECTURA DEL SISTEMA

```
[OpenWebUI :3000] ──────────────────────────────────────────┐
                                          host.docker.internal│
[Ollama :11434]  ──┐                                        │
[ChromaDB :8000] ──┼──▶  [Backend FastAPI :8080]  ◀─────────┘
[Groq API cloud] ──┘         (Python local, venv)
```

**¿Por qué el backend corre local y no en Docker?** Más fácil de debuggear, recarga automática
con `--reload`, no requiere rebuild de imagen por cada cambio.

**OpenWebUI no sabe que no es OpenAI.** Cree que está hablando con la API de OpenAI. El backend
expone el modelo `docker-rag-assistant` vía endpoints compatibles con OpenAI.

---

## 3. ESTRUCTURA DE DIRECTORIOS

```
ProyectoRagSoporteTecnico/
├── docker-compose.yml          # Ollama + ChromaDB + OpenWebUI
├── docker-compose.gpu.yml      # Override para GPU NVIDIA
├── .env / .env.example         # Variables de entorno
├── Makefile                    # Comandos simplificados
├── README.md                   # Documentación para el equipo
├── CAMBIOS.md                  # Historial de cambios implementados
├── AGENT.md                    # Este archivo
│
├── backend/
│   ├── requirements.txt        # Deps Python prod (Python 3.14)
│   ├── requirements-eval.txt   # Deps evaluación (Python 3.11 obligatorio)
│   ├── venv/                   # Entorno virtual prod (gitignored)
│   ├── venv-eval/              # Entorno virtual eval (gitignored)
│   │
│   ├── app/
│   │   ├── main.py             # FastAPI: endpoints /query, /v1/chat/completions, /health
│   │   ├── config.py           # Settings vía pydantic-settings, carga .env
│   │   ├── models.py           # Modelos Pydantic: RAGResponse, EvidenceResult, etc.
│   │   ├── pipeline.py         # ⭐ Orquestador principal del pipeline completo
│   │   │
│   │   ├── ingestion/          # Carga y procesamiento del corpus
│   │   │   ├── loader.py       # Lee archivos .md del corpus
│   │   │   ├── preprocessor.py # Limpieza de texto
│   │   │   ├── chunker.py      # División en chunks (800 chars, 200 overlap)
│   │   │   ├── metadata.py     # Enriquece metadata (categoría, plataforma, etc.)
│   │   │   └── run.py          # Script de ingesta: python -m app.ingestion.run
│   │   │
│   │   ├── retrieval/
│   │   │   ├── vector_store.py # ChromaDB client, get_semantic_retriever()
│   │   │   ├── bm25_retriever.py # BM25 se reconstruye desde ChromaDB al iniciar
│   │   │   ├── hybrid.py       # RRF (Reciprocal Rank Fusion): sem 0.4 + BM25 0.6
│   │   │   └── reranker.py     # FlashRank, top_n=10
│   │   │
│   │   └── generation/
│   │       ├── router.py       # ⭐ Semantic Router: CHITCHAT vs RAG (ver §6)
│   │       ├── evidence_checker.py # Decide si hay suficiente evidencia (ver §7)
│   │       ├── prompt_templates.py # Prompts especializados para Docker
│   │       ├── translator.py   # Detección ES/EN y traducción híbrida (ver §8)
│   │       └── generator.py    # Invoca LLM, parsea citas [Source N]
│   │
│   └── tests/
│       ├── eval_dataset.json   # 20 preguntas con ground_truth para evaluación
│       ├── test_evaluation.py  # ⭐ Pipeline RAGAS completo (ver §9)
│       ├── test_ingestion.py
│       └── test_vector_store.py
│
└── corpus/
    ├── raw/                    # Markdown crudo (gitignored)
    ├── processed/              # Markdown limpio (input de ingesta)
    └── scripts/
        ├── download_docs.py    # Descarga docs de GitHub
        └── prepare_corpus.py  # Preprocesamiento
```

---

## 4. PIPELINE COMPLETO (flujo de `pipeline.py`)

Cuando llega una pregunta vía `/v1/chat/completions`:

```
Pregunta usuario
      │
      ▼
[0] Detección idioma (langdetect, local, 0 tokens)
      │
      ├─ ES → Traducción ES→EN (Google Translate, gratis, 0 tokens LLM)
      │
      ▼
[1] Semantic Router (router.py)
      │
      ├─ CHITCHAT → Respuesta conversacional amigable (LLM directo, sin retrieval)
      │             RAGResponse con status="chit-chat", sources=[], chunks_used=0
      │
      └─ RAG → continúa ↓
              │
              ▼
[2] Retrieval Híbrido (top_k=30)
    BM25 (0.6) + Semántico ChromaDB (0.4) via RRF
              │
              ▼
[3] Reranking — FlashRank, top_n=10
              │
              ▼
[4] Evidence Check (evidence_checker.py)
    Señales: top_score, relevant_count, score_spread
              │
              ├─ INSUFFICIENT → Abstención (mensaje predefinido, no alucina)
              │
              └─ SUFFICIENT / LOW_CONFIDENCE → continúa ↓
                          │
                          ▼
              [5] Generación (generator.py)
                  LLM Groq llama-3.3-70b-versatile, temp=0.0
                  Parsea citas [Source N] del texto
                          │
                          ▼
              [6] Traducción EN→ES (si pregunta era español)
                  LLM Groq (preserva código y comandos Docker)
                          │
                          ▼
                   RAGResponse completo
```

---

## 5. PIPELINE DE INGESTA (`app/ingestion/`)

Orquestado por `run.py`. Se corre **una sola vez** (o cuando cambia el corpus).
Los embeddings persisten en el volumen Docker de ChromaDB.

```powershell
# Comando
python -m app.ingestion.run           # Normal (omite si ChromaDB ya tiene datos)
python -m app.ingestion.run --force   # Borra todo y re-indexa desde cero
```

### Paso 1 — Carga (`loader.py`)
Lee todos los archivos `.md` del directorio `corpus/processed/` de forma recursiva.
Devuelve una lista de `Document` de LangChain con el `page_content` (texto raw)
y `metadata["source_file"]` (path relativo del archivo).

### Paso 2 — Preprocesado (`preprocessor.py`)
Limpia el texto de cada documento antes de chunkear:
- Elimina shortcodes de Hugo (`{{< ... >}}`, `{{% ... %}}`)
- Elimina tags HTML residuales
- Extrae y preserva el frontmatter YAML (`title:`, `description:`, etc.)
- Descarta documentos que quedan vacíos o muy cortos tras la limpieza

### Paso 3 — Chunking (`chunker.py`)
Usa `RecursiveCharacterTextSplitter` de LangChain con separadores **Markdown-aware**,
ordenados de más grueso a más fino:

```python
MARKDOWN_SEPARATORS = [
    "\n## ",   # H2 — secciones principales (primer intento)
    "\n### ",  # H3
    "\n#### ", # H4
    "\n```",   # Bloques de código (se preservan como unidad)
    "\n\n",    # Párrafos
    "\n",      # Líneas
    ". ",      # Oraciones (último recurso antes de cortar palabras)
    " ",       # Palabras (último último recurso)
]
```

**Parámetros:** `chunk_size=800 chars`, `chunk_overlap=200 chars`, `keep_separator=True`
(el encabezado `## Foo` queda al inicio del chunk para que el LLM tenga contexto de sección).

Post-split, se agrega a la metadata de cada chunk:
- `chunk_index`: posición del chunk dentro de su documento fuente
- `total_chunks`: total de chunks del documento

**Resultado actual del corpus:** ~3211 chunks de ~3200+ archivos `.md`

### Paso 4 — Enriquecimiento de Metadata (`metadata.py`)

Cada chunk recibe 5 campos adicionales en su `metadata` dict:

| Campo | Cómo se detecta | Ejemplo |
|:---|:---|:---|
| `category` | Keywords en el **path** del archivo (prioridad) o en el contenido | `installation`, `troubleshooting`, `cli_reference` |
| `platform` | Keywords en path o frecuencia ≥3 en contenido | `windows`, `linux`, `mac`, `general` |
| `doc_title` | Del frontmatter YAML `title:` o del nombre del archivo | `"Windows Install"` |
| `section_header` | Primer encabezado `##`/`###`/`####` dentro del chunk | `"Installation methods"` |
| `doc_type` | Derivado de la categoría | `guide`, `reference`, `troubleshooting` |

**Reglas de categoría** (busca en orden, primer match gana):
```
"install", "setup"          → INSTALLATION
"get-started", "concepts"   → GETTING_STARTED  
"troubleshoot", "known-issues" → TROUBLESHOOTING
"reference", "cli"          → CLI_REFERENCE
"config", "daemon"          → CONFIGURATION
```

### Paso 5 — Indexado en ChromaDB (`retrieval/vector_store.py`)

```python
vector_store.add_documents(chunks)
# Internamente: OllamaEmbeddings(model="nomic-embed-text") genera el vector por chunk
# y ChromaDB lo persiste en su volumen Docker.
```

**Lógica de re-indexación:**
- Si la colección ya tiene datos y `--force` no está activo → **skip** (no re-indexa).
- Con `--force` → borra la colección entera y re-indexa desde cero.
- El BM25 **no** se persiste; se reconstruye en memoria cada vez que `RAGPipeline.__init__()` corre
  (carga los documentos directamente desde ChromaDB: `vector_store._collection.get(...)`).

---

## 6. MODELOS PYDANTIC CLAVE (`models.py`)

```python
RAGResponse
  ├── answer: str               # Texto de respuesta con citas inline [Source N]
  ├── sources: list[SourceCitation]
  ├── evidence: EvidenceResult  # verdict, top_score, relevant_count, details
  └── retrieval_metadata: RetrievalMetadata
        ├── question: str
        ├── original_question: str | None
        ├── translated_question: str | None
        ├── status: str | None  # "chit-chat" | "abstained" | None
        ├── chunks_used: int
        └── chunks_metadata: list[ChunkMetadata]

EvidenceVerdict (Enum): SUFFICIENT | LOW_CONFIDENCE | INSUFFICIENT
```

---

## 7. SEMANTIC ROUTER (`generation/router.py`)

Clasifica la intención ANTES de hacer retrieval. Estrategia en cascada:

**Etapa 1 — Heurísticas locales (0 tokens, ~0ms):**
- Keywords técnicas (`docker`, `compose`, `install`, `container`, etc.) → `RAG` inmediato
- Regex de saludos/agradecimientos (`Hola!`, `Gracias`, `Bye`, emojis solos) → `CHITCHAT`

**Etapa 2 — LLM few-shot (fallback, ~0.3s Groq):**
- Prompt ultra-compacto, responde estrictamente `RAG` o `CHITCHAT`
- Default a `RAG` ante cualquier ambigüedad (mejor buscar de más que ignorar pregunta técnica)

**Rutas:**
- `CHITCHAT` → `get_chitchat_response()`: respuesta amigable, idioma del usuario, sin retrieval
- `RAG` → pipeline completo

**Nota importante:** OpenWebUI genera requests automáticos (`### Task: Generate tags...`).
Esto se resolvió desactivando "AutoGeneración de Etiquetas" en la configuración de OpenWebUI,
no en código (el router en código quedó solo con RAG/CHITCHAT).

---

## 8. EVIDENCE CHECKER (`generation/evidence_checker.py`)

Evalúa si los chunks recuperados tienen suficiente información antes de generar.
Usa tres señales combinadas:

| Señal | Lógica | Resultado |
|:---|:---|:---|
| `top_score < min_top_score (0.3)` | El mejor chunk es muy malo | `INSUFFICIENT` |
| `score_spread < 0.05 AND top_score < 0.85` | Todos iguales de malos (ruido) | `LOW_CONFIDENCE` |
| `relevant_count < min_relevant_chunks (2)` | Muy poca información | `LOW_CONFIDENCE` |
| Pasa todos | Evidencia sólida | `SUFFICIENT` |

`LOW_CONFIDENCE` continúa al generador (responde con advertencia implícita).
`INSUFFICIENT` dispara la abstención (mensaje predefinido que no alucina).

**Configuración en `.env` / `config.py`:**
```
min_top_score = 0.3
min_relevant_chunks = 2
relevance_threshold = 0.25
```

---

## 9. TRADUCCIÓN MULTILINGÜE (`generation/translator.py`)

Estrategia híbrida que minimiza tokens LLM:

| Paso | Herramienta | Costo |
|:---|:---|:---|
| Detección idioma | `langdetect` (local) | 0 tokens |
| Query ES→EN | `deep-translator` (Google Translate) | 0 tokens |
| Respuesta EN→ES | LLM Groq | ~1 llamada LLM |

**¿Por qué usar LLM para traducir la respuesta (no Google Translate)?**
La respuesta contiene bloques de código (`docker run -d nginx`), paths, variables de entorno.
Google Translate los destruye. El LLM entiende el contexto y los preserva intactos.

El LLM ignorando el parámetro `llm` en `detect_spanish()` — se mantiene solo por compatibilidad
de firma; internamente ya no se usa (langdetect es determinístico y local).

---

## 10. EVALUACIÓN RAGAS (`tests/test_evaluation.py`)

**IMPORTANTE: Requiere `venv-eval` con Python 3.11.** RAGAS no es compatible con Python 3.14.

```powershell
# Activar entorno de evaluación
backend\venv-eval\Scripts\Activate.ps1
cd backend

# Correr evaluación quick (10 preguntas, Groq para generación + Ollama para evaluación)
python -m tests.test_evaluation --model llama-3.1-8b-instant --quick

# Evaluación completa (20 preguntas)
python -m tests.test_evaluation --model llama-3.3-70b-versatile
```

**Configuración del juez RAGAS:**
- **Juez LLM:** Ollama local (`llama3.1:8b` en `http://127.0.0.1:11434/v1`, OpenAI-compatible)
- **Embeddings:** `OllamaEmbeddings(model="nomic-embed-text")` + `LangchainEmbeddingsWrapper`
- **Instanciación:** `llm_factory()` con cliente `openai.OpenAI` apuntando a Ollama
- **Workers:** `RunConfig(max_workers=1)` — serializado para estabilidad

**Métricas evaluadas:**
| Métrica | Qué mide |
|:---|:---|
| `Faithfulness` | ¿La respuesta está soportada por los chunks? |
| `AnswerRelevancy` | ¿La respuesta es relevante a la pregunta? |
| `ContextPrecision` | ¿Los chunks recuperados son precisos? |
| `ContextRecall` | ¿Se recuperó toda la información necesaria? |

**Resultado del dry-run de validación:**
```
faithfulness:      0.6667
answer_relevancy:  0.8037
context_precision: 1.0000
context_recall:    1.0000
```

**¿Por qué Ollama como juez y no Groq?**
Groq tiene rate limits severos. Evaluar 20 preguntas × 4 métricas = 80+ llamadas consecutivas
→ errores 429. Ollama es local y no tiene límites.

---

## 11. CONFIGURACIÓN (`app/config.py`)

```python
# Ollama (local)
ollama_base_url = "http://127.0.0.1:11434"
llm_model = "llama3.1:8b"
embedding_model = "nomic-embed-text"

# Groq (nube) — REQUERIDO en .env
groq_api_key = "..."       # Variable de entorno GROQ_API_KEY

# ChromaDB
chroma_host = "localhost"
chroma_port = 8000
chroma_collection_name = "docker_docs"

# Chunking
chunk_size = 800
chunk_overlap = 200

# Retrieval
top_k = 30
rerank_top_n = 10
semantic_weight = 0.4
bm25_weight = 0.6

# Evidence Check
min_top_score = 0.3
min_relevant_chunks = 2
relevance_threshold = 0.25
```

---

## 12. COMANDOS DE TRABAJO DIARIO

```powershell
# === INFRAESTRUCTURA ===
docker compose up -d                    # Levantar Ollama + ChromaDB + OpenWebUI
docker compose down                     # Detener todo

# === BACKEND (venv de producción) ===
backend\venv\Scripts\Activate.ps1
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# === INGESTA (solo si cambia el corpus) ===
python -m app.ingestion.run

# === EVALUACIÓN (venv-eval, Python 3.11) ===
backend\venv-eval\Scripts\Activate.ps1
cd backend
python -m tests.test_evaluation --model llama-3.1-8b-instant --quick
```

---

## 13. GIT / RAMAS

- **Rama activa:** `integracion-webui`
- **`.gitignore` configurado** para excluir: `venv/`, `venv-eval/`, `*.pyc`, `__pycache__/`
- Los vectores de ChromaDB y los modelos de Ollama son persistidos en volúmenes Docker,
  no se versiona el corpus procesado

---

## 14. ENDPOINTS API

| Endpoint | Método | Descripción |
|:---|:---|:---|
| `/health` | GET | Health check (devuelve modelo activo) |
| `/docs` | GET | Swagger UI |
| `/query` | POST | Query directo al pipeline RAG |
| `/ingest` | POST | Re-indexar corpus |
| `/v1/models` | GET | Lista modelos (OpenAI-compatible) |
| `/v1/chat/completions` | POST | Chat completions (OpenAI-compatible, usado por OpenWebUI) |

OpenWebUI llama a `/v1/chat/completions`. El backend extrae el último mensaje con `role=user`
y lo pasa al `RAGPipeline.run()`.

---

## 15. FEATURES IMPLEMENTADAS (HISTORIAL)

| Feature | Estado | Archivo principal |
|:---|:---|:---|
| Ingesta y chunking de corpus Docker | ✅ | `app/ingestion/` |
| Retrieval híbrido BM25 + Vector (RRF) | ✅ | `app/retrieval/hybrid.py` |
| Reranking con FlashRank | ✅ | `app/retrieval/reranker.py` |
| Evidence Checker (anti-alucinación) | ✅ | `app/generation/evidence_checker.py` |
| Generación con citas [Source N] | ✅ | `app/generation/generator.py` |
| Compatibilidad OpenAI API (OpenWebUI) | ✅ | `app/main.py` |
| Traducción automática ES↔EN | ✅ | `app/generation/translator.py` |
| Semantic Router (CHITCHAT vs RAG) | ✅ | `app/generation/router.py` |
| Evaluación RAGAS con juez Ollama local | ✅ | `tests/test_evaluation.py` |

---

## 16. PRÓXIMAS FEATURES PLANIFICADAS

| Feature | Prioridad | Descripción |
|:---|:---|:---|
| **Historial de Conversación** | 🔥 Alta | Memoria de sesión. Reescribir preguntas dependientes ("¿y cómo lo hago en Linux?") en preguntas standalone antes del retrieval usando un LLM rápido. |
| **Caché Semántico** | Media | Redis o en memoria. Si pregunta similar (>95% similitud) fue respondida hace <2h, devolver respuesta cacheada sin retrieval. |
| **Streaming SSE** | Media | Server-Sent Events en `/v1/chat/completions`. El usuario vería la respuesta token a token como ChatGPT. |

---

## 17. PROBLEMAS CONOCIDOS / GOTCHAS

| Problema | Causa | Solución |
|:---|:---|:---|
| RAGAS no instala en Python 3.14 | `scikit-network` incompatible | Usar `venv-eval` con `py -3.11` |
| OpenWebUI manda requests automáticos de tags | Configuración de OpenWebUI | Desactivar "AutoGeneración de Etiquetas" en settings de OpenWebUI |
| `langdetect` detecta español rioplatense como `pt` (portugués) | Similitud ES/PT | El router LLM corrige correctamente; `langdetect` solo es primera barrera |
| Primera query tarda ~10s | Inicialización lazy del pipeline (BM25 + ChromaDB + FlashRank) | Normal. Desde la segunda query es ~2-5s |
| Groq 429 en evaluación | Rate limits de la API en tier gratuito | Usar Ollama como juez (ya implementado) |

---

## 18. VARIABLES DE ENTORNO REQUERIDAS (`.env`)

```env
GROQ_API_KEY=gsk_...           # Obligatorio para backend en producción
CHROMA_HOST=localhost           # Default
CHROMA_PORT=8000                # Default
OLLAMA_BASE_URL=http://127.0.0.1:11434  # Default
```

El archivo `.env` va en la raíz del proyecto o en `backend/`. `config.py` busca ambas rutas.

---

*Última actualización: 2026-05-01 | Branch: integracion-webui*
