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

    # TODO: Implementar evaluación multi-señal

    # Placeholder: siempre dice suficiente
    return EvidenceResult(
        verdict=EvidenceVerdict.SUFFICIENT,
        top_score=0.0,
        relevant_count=len(reranked_chunks),
        details="Evaluación de evidencia pendiente de implementación",
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
