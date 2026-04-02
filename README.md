# 🐳 Asistente RAG — Soporte Técnico Documental de Docker

Asistente conversacional especializado en soporte técnico de Docker, basado en documentación oficial. Utiliza Retrieval-Augmented Generation (RAG) con búsqueda híbrida, reranking, verificación de evidencia y citas trazables.

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Arquitectura](#arquitectura)
- [Stack Tecnológico](#stack-tecnológico)
- [Requisitos Previos](#requisitos-previos)
- [Instalación y Setup](#instalación-y-setup)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Notas para el Equipo](#notas-para-el-equipo)

---

## Descripción

Este sistema responde preguntas sobre Docker (instalación, uso inicial, troubleshooting) fundamentando sus respuestas **exclusivamente en documentación oficial**, con citas verificables y un detector de evidencia insuficiente para evitar alucinaciones.

**No es un RAG genérico.** El corpus, los prompts, la metadata y la evaluación están diseñados específicamente para soporte técnico documental de Docker.

---

## Arquitectura

El proyecto separa **infraestructura** (Docker) y **lógica de negocio** (Python local):

```
┌─────────────────────────────────────────────────────────────┐
│                     EN DOCKER                               │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Ollama      │  │  ChromaDB    │  │  OpenWebUI   │      │
│  │  LLM + Embed  │  │  Vectores    │  │  Frontend    │      │
│  │  :11434       │  │  :8000       │  │  :3000       │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                  │              │
└─────────┼─────────────────┼──────────────────┼──────────────┘
          │                 │                  │
          │    localhost     │    localhost      │  host.docker.internal
          │                 │                  │
┌─────────┼─────────────────┼──────────────────┼──────────────┐
│         │                 │                  │              │
│  ┌──────▼─────────────────▼──────────────────▼───────────┐  │
│  │           Backend RAG (FastAPI + LangChain)            │  │
│  │              Python local en venv — :8080              │  │
│  │                                                       │  │
│  │  Ingesta │ Retrieval Híbrido │ Reranking │ Generación  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│                     LOCAL (venv Python)                      │
└─────────────────────────────────────────────────────────────┘
```

**¿Por qué el backend corre local y no en Docker?**
- Más fácil de debuggear y desarrollar
- Recarga automática con `--reload`
- No hay que hacer rebuild de imagen para cada cambio
- Los servicios pesados (LLM, base vectorial, frontend) sí van en Docker

---

## Stack Tecnológico

| Componente | Tecnología | Corre en |
|:---|:---|:---|
| Frontend | OpenWebUI | Docker |
| LLM | Ollama (llama3.1:8b) | Docker |
| Embeddings | Ollama (nomic-embed-text) | Docker |
| Base Vectorial | ChromaDB | Docker |
| Backend RAG | FastAPI + LangChain | **Local (venv)** |
| Búsqueda Léxica | BM25 (rank-bm25) | Local (venv) |
| Reranking | FlashRank | Local (venv) |
| Evaluación | LLM-as-Judge (custom) | Local (venv) |

---

## Requisitos Previos

### Para TODOS los miembros del equipo

- **Docker Desktop** instalado y funcionando
  - Windows: [Descargar Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)
  - Requiere WSL 2 habilitado en Windows
- **Git** instalado
- **Python 3.11+** instalado (compatible con Python 3.14)
- **Make** (opcional)
  - Windows: `choco install make` o `winget install GnuWin32.Make`
  - Si no lo tenés, podés usar los comandos directamente (ver sección "Sin Make")

### Solo si tenés GPU NVIDIA (opcional, mejora rendimiento)

- GPU NVIDIA con al menos 6 GB de VRAM (ej: RTX 2060)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) instalado
- Drivers NVIDIA actualizados

> **Sin GPU funciona igual.** Solo tarda más cada respuesta (~15-30s en CPU vs ~2-5s con GPU).

---

## Instalación y Setup

### Paso 1 — Clonar el repositorio

```bash
git clone <URL_DEL_REPO>
cd ProyectoRagSoporteTecnico
```

### Paso 2 — Configuración de entorno

```bash
copy .env.example .env
```

Revisá `.env` y ajustá puertos si hay conflictos (ej: si ya usás el 3000).

### Paso 3 — Crear el entorno virtual de Python

```powershell
# Crear venv
python -m venv backend\venv

# Activar (PowerShell)
backend\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Instalar dependencias
pip install -r backend\requirements.txt
```

### Paso 4 — Levantar servicios Docker (Ollama + ChromaDB + OpenWebUI)

**Sin GPU (la mayoría del equipo):**
```bash
docker compose up -d
```

**Con GPU NVIDIA:**
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### Paso 5 — Descargar modelos de Ollama (solo la primera vez)

```bash
docker exec rag-ollama ollama pull llama3.1:8b
docker exec rag-ollama ollama pull nomic-embed-text
```

> ⏳ Puede tardar varios minutos. Los modelos se guardan en un volumen Docker persistente (se descargan una sola vez).

### Paso 6 — Correr el backend local

Con el venv activado:

```powershell
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### Paso 7 — Ingestar el corpus (solo la primera vez)

Asegurate de tener los archivos `.md` en `corpus/processed/`, luego con el venv activado:

```powershell
cd backend
python -m app.ingestion.run
```

> Los embeddings se guardan en ChromaDB (volumen Docker). **No se re-indexan cada vez que levantás el proyecto.** Solo hace falta si actualizás el corpus.

### Paso 8 — Usar el asistente

Abrí en tu navegador:

```
http://localhost:3000
```

---

## Uso

### Comandos principales (con Make)

| Comando | Descripción |
|:---|:---|
| `make up` | Levantar Docker (CPU) |
| `make up-gpu` | Levantar Docker (GPU NVIDIA) |
| `make down` | Detener Docker |
| `make pull-models` | Descargar modelos Ollama |
| `make setup-venv` | Crear venv + instalar dependencias |
| `make run-backend` | Correr backend FastAPI local |
| `make ingest` | Indexar corpus en ChromaDB |
| `make evaluate` | Ejecutar evaluación comparativa |
| `make logs` | Ver logs de Docker |
| `make clean` | Reset completo (¡borra todo!) |

### Sin Make (comandos directos)

```powershell
# --- Docker (infraestructura) ---
docker compose up -d                                    # Levantar (CPU)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d  # Con GPU
docker compose down                                     # Detener
docker exec rag-ollama ollama pull llama3.1:8b          # Descargar LLM
docker exec rag-ollama ollama pull nomic-embed-text     # Descargar embeddings

# --- Backend local (con venv activado) ---
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload   # Correr backend
python -m app.ingestion.run                                  # Ingestar corpus
python -m evaluation.run_evaluation                          # Evaluar
```

### Endpoints del Backend

| Endpoint | Método | Descripción |
|:---|:---|:---|
| `/health` | GET | Health check |
| `/docs` | GET | Documentación Swagger |
| `/query` | POST | Consulta al pipeline RAG |
| `/ingest` | POST | Re-indexar corpus |
| `/v1/models` | GET | Modelos (OpenAI-compatible) |
| `/v1/chat/completions` | POST | Chat (OpenAI-compatible) |

---

## Estructura del Proyecto

```
ProyectoRagSoporteTecnico/
├── docker-compose.yml           # Ollama + ChromaDB + OpenWebUI
├── docker-compose.gpu.yml       # Override GPU NVIDIA
├── .env / .env.example          # Variables de entorno
├── Makefile                     # Comandos simplificados
├── README.md
│
├── backend/                     # 🔧 Backend RAG (corre LOCAL en venv)
│   ├── requirements.txt         # Dependencias Python
│   ├── app/
│   │   ├── main.py              # FastAPI endpoints
│   │   ├── config.py            # Configuración centralizada
│   │   ├── models.py            # Modelos Pydantic
│   │   ├── pipeline.py          # Orquestador del pipeline RAG
│   │   │
│   │   ├── ingestion/           # Carga y procesamiento del corpus
│   │   │   ├── loader.py        # Carga archivos Markdown
│   │   │   ├── preprocessor.py  # Limpieza de texto
│   │   │   ├── chunker.py       # División en chunks
│   │   │   ├── metadata.py      # Enriquecimiento de metadata
│   │   │   └── run.py           # Script de ingesta completo
│   │   │
│   │   ├── retrieval/           # Búsqueda y recuperación
│   │   │   ├── vector_store.py  # ChromaDB (semántico)
│   │   │   ├── bm25_retriever.py# BM25 (léxico)
│   │   │   ├── hybrid.py        # Retriever híbrido (RRF)
│   │   │   └── reranker.py      # FlashRank reranking
│   │   │
│   │   └── generation/          # Generación de respuestas
│   │       ├── evidence_checker.py # Detector de evidencia
│   │       ├── prompt_templates.py # Prompts especializados
│   │       └── generator.py     # Invocación LLM + citas
│   │
│   └── tests/                   # Tests unitarios
│       ├── test_chunker.py
│       ├── test_retrieval.py
│       └── test_evidence.py
│
├── corpus/                      # 📄 Documentación Docker
│   ├── raw/                     # Markdown crudo (gitignored)
│   ├── processed/               # Markdown limpio (input de ingesta)
│   └── scripts/
│       ├── download_docs.py     # Descarga docs de GitHub
│       └── prepare_corpus.py    # Preprocesamiento
│
├── evaluation/                  # 📊 Evaluación (LLM-as-Judge)
│   ├── test_questions.json      # Preguntas de test
│   ├── run_evaluation.py        # Script de evaluación
│   ├── metrics.py               # Métricas con LLM como juez
│   └── results/                 # Resultados por variante
│
└── docs/                        # 📖 Documentación del proyecto
    ├── architecture.md          # Diagrama de arquitectura
    └── prompts.md               # Documentación de prompts
```

---

## Notas para el Equipo

### Persistencia de datos

| Dato | ¿Persiste? | ¿Dónde? | ¿Cuándo se pierde? |
|:---|:---|:---|:---|
| Modelos Ollama | ✅ Sí | Volumen Docker `rag_ollama_data` | Solo con `docker compose down -v` |
| Embeddings (ChromaDB) | ✅ Sí | Volumen Docker `rag_chroma_data` | Solo con `docker compose down -v` |
| Config OpenWebUI | ✅ Sí | Volumen Docker `rag_openwebui_data` | Solo con `docker compose down -v` |
| Código Python | 📁 Local | `backend/` en tu repo | No se pierde |
| venv | 📁 Local | `backend/venv/` (gitignored) | Si borrás la carpeta |

> **Los embeddings NO se recalculan** cada vez que levantás el proyecto. Solo ejecutar `python -m app.ingestion.run` la primera vez o si cambia el corpus.

### Con GPU vs Sin GPU

- **Con GPU (ej: RTX 2060):** ~2-5 segundos por respuesta. Usar `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`
- **Sin GPU (CPU):** ~15-30 segundos por respuesta. Usar `docker compose up -d`
- El modelo `llama3.1:8b` usa ~5 GB de VRAM (GPU) o RAM (CPU)

### Flujo de trabajo diario

```powershell
git pull                              # 1. Traer cambios
docker compose up -d                  # 2. Levantar infra (si no está corriendo)
backend\venv\Scripts\Activate.ps1     # 3. Activar venv
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload  # 4. Correr backend
# ... desarrollar ...
# Ctrl+C para detener el backend
docker compose down                   # 5. Al terminar (opcional)
```

### Troubleshooting

**"Cannot connect to Docker daemon"**
→ Asegurate de que Docker Desktop esté corriendo.

**"Port 3000 already in use"**
→ Cambiá `WEBUI_PORT=3001` en tu `.env`.

**Ollama tarda mucho en responder**
→ Normal en CPU. La primera consulta carga el modelo en memoria, las siguientes son más rápidas.

**`make` no se reconoce como comando**
→ `choco install make` o usá los comandos directamente (ver "Sin Make").

**OpenWebUI no se conecta al backend**
→ Verificá que el backend esté corriendo en el puerto 8080. OpenWebUI lo busca en `host.docker.internal:8080`.

**Error al instalar dependencias Python**
→ Asegurate de tener el venv activado. Si ves errores de compilación, puede faltar [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
