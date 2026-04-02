"""
Backend RAG — FastAPI Application

Punto de entrada del backend. Define los endpoints HTTP
que OpenWebUI (vía Pipelines) y el frontend consumen.
"""

from fastapi import FastAPI, BackgroundTasks
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

from functools import lru_cache

@lru_cache()
def get_pipeline():
    from app.pipeline import RAGPipeline
    return RAGPipeline()

@app.post("/query")
async def query_rag(question: str):
    """
    Endpoint principal del pipeline RAG.

    Recibe una pregunta en lenguaje natural y devuelve una respuesta
    fundamentada en la documentación de Docker, con citas y trazabilidad.
    """
    try:
        pipeline = get_pipeline()
        response = pipeline.run(question)
        return response.model_dump()
    except Exception as e:
        logger.error(f"Error procesando query: {e}")
        return {"error": str(e)}


@app.post("/ingest")
async def ingest_corpus(background_tasks: BackgroundTasks, force: bool = False):
    """
    Trigger manual para re-indexar el corpus en ChromaDB.

    Solo necesario si se actualiza la documentación fuente.
    Los embeddings persisten en el volumen Docker de ChromaDB.
    
    La ejecución se realiza en 2do plano porque generar los embeddings locales
    puede tomar varios minutos.
    """
    from app.ingestion.run import run_ingestion
    
    # Agregar la función al spooler de tareas en segundo plano de FastAPI
    background_tasks.add_task(run_ingestion, force=force)
    
    return {
        "status": "processing",
        "message": "Ingesta iniciada en segundo plano. Esto puede tomar varios minutos dependiendo de la máquina.",
        "force_applied": force
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
