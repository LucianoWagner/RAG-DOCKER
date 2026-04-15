"""
Generator — Invocación del LLM y armado de respuesta con citas.

Responsabilidad:
- Invocar el LLM (Ollama) con el prompt construido
- Parsear la respuesta para extraer citas [Fuente N]
- Construir el objeto RAGResponse con respuesta + citas + metadata

Configuración importante de Groq:
- El modelo se define en el constructor de ChatGroq.
"""

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from loguru import logger

from app.config import get_settings
from app.models import RAGResponse, SourceCitation, EvidenceResult, RetrievalMetadata, ChunkMetadata


def get_llm() -> ChatGroq:
    """
    Crea la instancia del LLM vía la nube súper-rápida de Groq.

    Returns:
        ChatGroq configurado. Lee GROQ_API_KEY desde configuración.
    """
    settings = get_settings()
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY no encontrada. Agregala a tu archivo .env")
        
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",  # Modelo ligero para bypass de tier de Groq
        api_key=settings.groq_api_key,
        temperature=0.0,  # Temperatura CERO: Robot puro copista/pegador, 0% creatividad y sustracción de alucinaciones
    )

def generate_response(
    question: str,
    context_chunks: list[Document],
    evidence: EvidenceResult,
    messages: list[dict],
    llm: ChatGroq | None = None,
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
    if llm is None:
        llm = get_llm()  # Fallback si se llama sin instancia (e.g. tests)
    logger.info(f"Generando respuesta | chunks: {len(context_chunks)}")

    # Invocación sincrónica cruda a Ollama
    response = llm.invoke(messages)
    answer_text = response.content

    # Mapeo de citas post-generación
    citations = parse_citations(answer_text, context_chunks)

    return RAGResponse(
        answer=answer_text,
        sources=citations,
        evidence=evidence,
        retrieval_metadata=RetrievalMetadata(
            question=question,
            chunks_used=len(context_chunks),
            chunks_metadata=[
                ChunkMetadata(**chunk.metadata, chunk_text=chunk.page_content) for chunk in context_chunks
            ]
        ),
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
    import re
    # Busca todas las citas de estilo [Source 1] [Source 2] ...
    matches = re.findall(r"\[Source (\d+)\]", answer_text, re.IGNORECASE)
    
    citations = []
    seen = set()
    for num_str in matches:
        try:
            n = int(num_str)
            index = n - 1
            if 0 <= index < len(chunks) and index not in seen:
                seen.add(index)
                chunk = chunks[index]
                meta = chunk.metadata
                citations.append(
                    SourceCitation(
                        citation_id=n,
                        source_file=meta.get("source_file", "unknown"),
                        doc_title=meta.get("doc_title", "Unspecified"),
                        section_header=meta.get("section_header", "General"),
                        relevant_fragment=chunk.page_content,
                        relevance_score=meta.get("relevance_score", 0.0)
                    )
                )
        except Exception as e:
            logger.warning(f"Error parseando cita [Fuente {num_str}]: {e}")
            continue
            
    return citations
