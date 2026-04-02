"""
Backend RAG — FastAPI Application

Punto de entrada del backend. Define los endpoints HTTP
que OpenWebUI (vía Pipelines) y el frontend consumen.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import get_settings

# =============================================================================
# Inicialización de la app
# =============================================================================

settings = get_settings()

app = FastAPI(
    title="RAG Docker Support API",
    description="Asistente RAG para soporte técnico documental de Docker",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"Backend RAG iniciado | LLM: {settings.llm_model} | Embeddings: {settings.embedding_model}")


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
async def health_check():
    """Verifica que el backend está corriendo."""
    return {
        "status": "ok",
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
    }


# =============================================================================
# Endpoints RAG
# =============================================================================

@app.post("/query")
async def query_rag(question: str):
    """
    Endpoint principal del pipeline RAG.

    Recibe una pregunta en lenguaje natural y devuelve una respuesta
    fundamentada en la documentación de Docker, con citas y trazabilidad.

    TODO: Implementar — conectar con pipeline.py
    """
    # TODO: Integrar con RAGPipeline
    return {
        "status": "not_implemented",
        "message": "Pipeline RAG pendiente de implementación",
        "question": question,
    }


@app.post("/ingest")
async def ingest_corpus():
    """
    Trigger manual para re-indexar el corpus en ChromaDB.

    Solo necesario si se actualiza la documentación fuente.
    Los embeddings persisten en el volumen Docker de ChromaDB.

    TODO: Implementar — conectar con ingestion/run.py
    """
    # TODO: Integrar con módulo de ingesta
    return {
        "status": "not_implemented",
        "message": "Ingesta pendiente de implementación",
    }


# =============================================================================
# Endpoints compatibles con OpenAI API (para OpenWebUI)
# =============================================================================

@app.get("/v1/models")
async def list_models():
    """
    Endpoint compatible con OpenAI API.
    OpenWebUI lo consulta para listar modelos disponibles.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "docker-rag-assistant",
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    """
    Endpoint compatible con OpenAI Chat Completions API.
    OpenWebUI envía las consultas aquí cuando se selecciona
    el modelo 'docker-rag-assistant'.

    TODO: Implementar — parsear request de OpenAI format,
          ejecutar pipeline RAG, devolver en formato OpenAI.
    """
    # TODO: Parsear mensajes del formato OpenAI
    # TODO: Extraer la pregunta del usuario
    # TODO: Ejecutar pipeline RAG
    # TODO: Formatear respuesta como OpenAI ChatCompletion

    # Respuesta placeholder en formato OpenAI
    return {
        "id": "chatcmpl-placeholder",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "🚧 Pipeline RAG en construcción. Este endpoint será implementado próximamente.",
                },
                "finish_reason": "stop",
            }
        ],
        "model": "docker-rag-assistant",
    }
