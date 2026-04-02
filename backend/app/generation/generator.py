"""
Generator — Invocación del LLM y armado de respuesta con citas.

Responsabilidad:
- Invocar el LLM (Ollama) con el prompt construido
- Parsear la respuesta para extraer citas [Fuente N]
- Construir el objeto RAGResponse con respuesta + citas + metadata

Configuración importante de Ollama:
- num_ctx: Context window. Default de Ollama es 2048 (MUY poco para RAG).
  Se debe configurar a 8192+ para que el LLM vea todos los chunks.
"""

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from loguru import logger

from app.config import get_settings
from app.models import RAGResponse, SourceCitation, EvidenceResult


def get_llm() -> ChatOllama:
    """
    Crea la instancia del LLM vía Ollama.

    Returns:
        ChatOllama configurado con modelo y context window adecuados.

    TODO: Implementar
    - ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        num_ctx=8192,  # CRÍTICO: aumentar context window
        temperature=0.1,  # Baja temperatura para respuestas precisas
      )
    """
    settings = get_settings()
    # TODO: Implementar
    raise NotImplementedError


def generate_response(
    question: str,
    context_chunks: list[Document],
    evidence: EvidenceResult,
    messages: list[dict],
) -> RAGResponse:
    """
    Genera la respuesta final del pipeline RAG.

    Args:
        question: Pregunta original del usuario.
        context_chunks: Chunks usados como contexto.
        evidence: Resultado de la verificación de evidencia.
        messages: Lista de mensajes formateados para el LLM.

    Returns:
        RAGResponse con respuesta, citas y metadata.

    TODO: Implementar
    1. Invocar el LLM con los mensajes
    2. Extraer texto de la respuesta
    3. Parsear citas [Fuente N] del texto
    4. Construir lista de SourceCitation con metadata de cada chunk citado
    5. Armar RAGResponse completo
    """
    logger.info(f"Generando respuesta | chunks: {len(context_chunks)}")

    # TODO: Implementar
    return RAGResponse(
        answer="Respuesta pendiente de implementación",
        sources=[],
        evidence=evidence,
        retrieval_metadata={
            "question": question,
            "chunks_used": len(context_chunks),
        },
    )


def parse_citations(answer_text: str, chunks: list[Document]) -> list[SourceCitation]:
    """
    Extrae las citas [Fuente N] del texto de respuesta y las vincula
    con los chunks correspondientes.

    Args:
        answer_text: Texto de respuesta del LLM con [Fuente N].
        chunks: Lista de chunks que se usaron como contexto.

    Returns:
        Lista de SourceCitation con la información de cada fuente citada.

    TODO: Implementar
    - Usar regex para encontrar todas las ocurrencias de [Fuente N]
    - Para cada N, obtener el chunk correspondiente (por índice)
    - Construir SourceCitation con metadata del chunk
    """
    # TODO: Implementar
    return []
