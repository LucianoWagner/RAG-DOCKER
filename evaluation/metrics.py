"""
Metrics — Evaluación RAG usando LLM-as-Judge.

En lugar de depender de RAGAS (que tiene problemas de compatibilidad
con Python 3.14), implementamos evaluación custom usando el mismo
LLM de Ollama como juez.

Esto tiene ventajas:
- Funciona con cualquier versión de Python
- No requiere dependencias externas adicionales
- Es más educativo (se entiende qué mide cada métrica)
- Usa el mismo stack del proyecto (Ollama)

Métricas implementadas:

Retriever:
- Context Precision: ¿los chunks recuperados son relevantes?
- Context Recall: ¿se recuperó toda la info necesaria?
- Hit Rate@K: ¿al menos un chunk del top-K es relevante?

Generator:
- Faithfulness: ¿la respuesta se basa solo en el contexto?
- Answer Relevancy: ¿la respuesta aborda la pregunta?

Sistema:
- Tasa de abstención: % de preguntas donde el sistema se abstiene
- Abstención correcta: % de abstenciones apropiadas
- Latencia: tiempo end-to-end
"""

from loguru import logger


# =============================================================================
# Prompts para LLM-as-Judge
# =============================================================================

FAITHFULNESS_JUDGE_PROMPT = """Evaluá si la respuesta se basa EXCLUSIVAMENTE en el contexto proporcionado.

CONTEXTO:
{context}

RESPUESTA:
{answer}

Respondé con un JSON:
{{"score": <0.0 a 1.0>, "explanation": "<breve justificación>"}}

- 1.0 = Toda la información de la respuesta está en el contexto
- 0.5 = Parte de la información está en el contexto, parte no
- 0.0 = La respuesta no se basa en el contexto
"""

ANSWER_RELEVANCY_JUDGE_PROMPT = """Evaluá si la respuesta aborda la pregunta del usuario.

PREGUNTA:
{question}

RESPUESTA:
{answer}

Respondé con un JSON:
{{"score": <0.0 a 1.0>, "explanation": "<breve justificación>"}}

- 1.0 = La respuesta aborda completamente la pregunta
- 0.5 = La respuesta aborda parcialmente la pregunta
- 0.0 = La respuesta no tiene relación con la pregunta
"""

CONTEXT_PRECISION_JUDGE_PROMPT = """Evaluá si los fragmentos de contexto recuperados son relevantes para responder la pregunta.

PREGUNTA:
{question}

FRAGMENTOS RECUPERADOS:
{contexts}

Para cada fragmento, indicá si es relevante (1) o no relevante (0).
Respondé con un JSON:
{{"relevant_count": <int>, "total_count": <int>, "score": <0.0 a 1.0>}}
"""


# =============================================================================
# Funciones de evaluación
# =============================================================================

def calculate_metrics(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """
    Calcula todas las métricas de evaluación usando LLM-as-Judge.

    Args:
        predictions: Lista de resultados del pipeline.
            Cada item: {"question", "answer", "contexts", "source_chunks"}
        ground_truth: Lista de respuestas esperadas.
            Cada item: {"question", "expected_answer", "expected_behavior"}

    Returns:
        Dict con todas las métricas calculadas.

    TODO: Implementar
    1. Para cada predicción, invocar el LLM con los prompts de juez
    2. Parsear los scores JSON de cada evaluación
    3. Calcular promedios por métrica
    4. Calcular métricas de sistema (abstención, latencia)
    5. Retornar dict consolidado
    """
    logger.info(f"Calculando métricas para {len(predictions)} predicciones...")

    # TODO: Implementar con LLM-as-Judge via Ollama

    metrics = {
        "retriever": {
            "context_precision": 0.0,
            "context_recall": 0.0,
            "hit_rate_at_k": 0.0,
        },
        "generator": {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
        },
        "system": {
            "abstention_rate": 0.0,
            "correct_abstention_rate": 0.0,
            "avg_latency_seconds": 0.0,
        },
    }

    return metrics


def evaluate_faithfulness(answer: str, context: str) -> dict:
    """
    Evalúa si la respuesta se basa en el contexto (detecta alucinaciones).

    TODO: Implementar
    - Formatear FAITHFULNESS_JUDGE_PROMPT con answer y context
    - Invocar LLM vía Ollama
    - Parsear JSON de respuesta
    - Retornar {"score": float, "explanation": str}
    """
    # TODO: Implementar
    return {"score": 0.0, "explanation": "Pendiente de implementación"}


def evaluate_answer_relevancy(question: str, answer: str) -> dict:
    """
    Evalúa si la respuesta aborda la pregunta del usuario.

    TODO: Implementar
    - Formatear ANSWER_RELEVANCY_JUDGE_PROMPT con question y answer
    - Invocar LLM vía Ollama
    - Parsear JSON de respuesta
    - Retornar {"score": float, "explanation": str}
    """
    # TODO: Implementar
    return {"score": 0.0, "explanation": "Pendiente de implementación"}


def evaluate_context_precision(question: str, contexts: list[str]) -> dict:
    """
    Evalúa si los chunks recuperados son relevantes para la pregunta.

    TODO: Implementar
    - Formatear CONTEXT_PRECISION_JUDGE_PROMPT con question y contexts
    - Invocar LLM vía Ollama
    - Parsear JSON de respuesta
    - Retornar {"score": float, "relevant_count": int, "total_count": int}
    """
    # TODO: Implementar
    return {"score": 0.0, "relevant_count": 0, "total_count": len(contexts)}
