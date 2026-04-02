"""
Evidence Checker — Detector de evidencia insuficiente.

Responsabilidad:
- Evaluar si los chunks rerankeados son suficientes para responder
- Usar un enfoque multi-señal (no solo score de embeddings)
- Decidir si el sistema debe responder o abstenerse

Señales evaluadas:
1. Score del top chunk (post-reranking)
2. Cantidad de chunks con score > umbral
3. Dispersión de scores (si todos son bajos y similares → ruido)
4. Cobertura temática (¿los chunks matchean la categoría de la pregunta?)

Esto es CRÍTICO para evitar alucinaciones: es mejor decir
"no tengo información" que inventar una respuesta incorrecta.
"""

from langchain_core.documents import Document
from loguru import logger

from app.config import get_settings
from app.models import EvidenceResult, EvidenceVerdict


def check_evidence(
    query: str,
    reranked_chunks: list[Document],
) -> EvidenceResult:
    """
    Evalúa si la evidencia recuperada es suficiente para responder.

    Args:
        query: Pregunta del usuario.
        reranked_chunks: Chunks rerankeados con rerank_score en metadata.

    Returns:
        EvidenceResult con veredicto (SUFFICIENT/LOW_CONFIDENCE/INSUFFICIENT).

    TODO: Implementar las señales de evaluación:

    Señal 1 — Score del mejor chunk:
    - Extraer rerank_score del top chunk
    - Si < settings.min_top_score → INSUFFICIENT

    Señal 2 — Cantidad de chunks relevantes:
    - Contar chunks con rerank_score > settings.relevance_threshold
    - Si < settings.min_relevant_chunks → LOW_CONFIDENCE

    Señal 3 — Dispersión de scores:
    - Calcular diferencia entre top 1 y último chunk
    - Si top_score < 0.4 Y spread < 0.1 → INSUFFICIENT
      (todos los scores bajos y similares = ninguno es realmente relevante)

    Señal 4 — Cobertura temática (opcional, mejora futura):
    - Verificar si la categoría de los chunks coincide con la pregunta
    """
    settings = get_settings()
    logger.info(f"Evaluando evidencia para: {query[:50]}...")

    if not reranked_chunks:
        return EvidenceResult(
            verdict=EvidenceVerdict.INSUFFICIENT,
            top_score=0.0,
            relevant_count=0,
            details="No se encontraron documentos recuperados."
        )

    # El reranker ya los ordenó, así que el primero es el mejor
    top_chunk = reranked_chunks[0]
    top_score = top_chunk.metadata.get("rerank_score", 0.0)

    # Contar cuántos chunks superan el umbral mínimo de relevancia
    relevant_chunks = [c for c in reranked_chunks if c.metadata.get("rerank_score", 0.0) >= settings.relevance_threshold]
    relevant_count = len(relevant_chunks)

    # Señal 1: El mejor resultado es objetivamente muy pobre
    if top_score < settings.min_top_score:
        logger.warning(f"Evidencia insuficiente: El top score ({top_score:.2f}) es menor al mínimo requerido ({settings.min_top_score})")
        return EvidenceResult(
            verdict=EvidenceVerdict.INSUFFICIENT,
            top_score=top_score,
            relevant_count=relevant_count,
            details=f"Mejor match tiene score {top_score:.2f} < {settings.min_top_score}"
        )

    # Señal 3: Dispersión de scores (Ruido general)
    last_score = reranked_chunks[-1].metadata.get("rerank_score", 0.0)
    score_spread = top_score - last_score
    if top_score < 0.85 and score_spread < 0.05:
        # Los scores son idénticos o casi idénticos y no son excepcionalmente altos.
        # Esto suele indicar que el modelo está "adivinando" porque no hay un ganador obvio.
        logger.warning(f"Baja confianza: Spread de scores muy bajo ({score_spread:.2f}) indicando probable ruido")
        return EvidenceResult(
            verdict=EvidenceVerdict.LOW_CONFIDENCE,
            top_score=top_score,
            relevant_count=relevant_count,
            details=f"Dispersión baja ({score_spread:.2f}) indica falta de información resolutiva."
        )

    # Señal 2: Tenemos información pero muy poca (menos de 2 chunks pasables)
    if relevant_count < settings.min_relevant_chunks:
        logger.warning(f"Baja confianza: Solo hay {relevant_count} chunks relevantes, esperábamos {settings.min_relevant_chunks}")
        return EvidenceResult(
            verdict=EvidenceVerdict.LOW_CONFIDENCE,
            top_score=top_score,
            relevant_count=relevant_count,
            details=f"Pocos chunks relevantes ({relevant_count} < {settings.min_relevant_chunks})"
        )

    # Si pasa todos los filtros, es SUFICIENTE
    logger.info(f"✅ Evidencia sólida: Top Score {top_score:.2f} con {relevant_count} chunks relevantes.")
    return EvidenceResult(
        verdict=EvidenceVerdict.SUFFICIENT,
        top_score=top_score,
        relevant_count=relevant_count,
        details="Evidencia aprobada por análisis de rank."
    )


def get_abstention_response() -> str:
    """
    Genera el mensaje de abstención cuando la evidencia es insuficiente.

    Returns:
        Texto de respuesta indicando que no hay información suficiente.
    """
    return (
        "⚠️ No encontré información suficiente en la documentación oficial de Docker "
        "para responder tu pregunta con confianza.\n\n"
        "Esto puede significar que:\n"
        "- La pregunta está fuera del alcance de la documentación indexada.\n"
        "- El tema requiere documentación más específica que no está en el corpus actual.\n\n"
        "💡 **Sugerencia:** Consultá directamente [docs.docker.com](https://docs.docker.com) "
        "o reformulá tu pregunta enfocándote en un aspecto más específico."
    )
