"""
Run Evaluation — Script principal de evaluación comparativa.

Ejecuta el pipeline RAG con distintas variantes de configuración
y calcula métricas de evaluación para compararlas.

Variantes definidas en el plan:
- V1: Vectorial puro (solo ChromaDB, sin BM25)
- V2: Híbrido (semántico + BM25)
- V3: Híbrido + Reranking
- V4: Híbrido + Reranking + Detector de evidencia
- V5: Chunk size 500
- V6: Chunk size 1200
- V7: Prompt restrictivo
- V8: Prompt permisivo

Métricas:
- Context Precision / Recall (retriever)
- Faithfulness (generator — ¿se basa en el contexto?)
- Answer Relevancy (generator — ¿responde la pregunta?)
- Tasa de abstención y abstención correcta
- Latencia end-to-end

Uso:
    python -m evaluation.run_evaluation
    o: make evaluate
"""

import json
from pathlib import Path

from loguru import logger


QUESTIONS_FILE = Path(__file__).parent / "test_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_test_questions() -> list[dict]:
    """Carga las preguntas de evaluación desde el JSON."""
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation():
    """
    Ejecuta la evaluación comparativa de variantes del pipeline.

    TODO: Implementar
    1. Cargar preguntas de test_questions.json
    2. Para cada variante (V1-V8):
       a. Configurar el pipeline con los parámetros de la variante
       b. Ejecutar cada pregunta del test set
       c. Registrar: respuesta, chunks recuperados, scores, tiempo
    3. Calcular métricas por variante
    4. Guardar resultados en evaluation/results/{variante}/
    5. Generar resumen comparativo

    Nota: Esto puede tomar bastante tiempo (8 variantes × N preguntas × LLM).
    Considerar ejecutar variantes individualmente.
    """
    logger.info("=" * 60)
    logger.info("EVALUACIÓN COMPARATIVA DEL PIPELINE RAG")
    logger.info("=" * 60)

    questions = load_test_questions()
    logger.info(f"Preguntas cargadas: {len(questions)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: Implementar loop de evaluación por variante
    logger.info("⚠️ Evaluación pendiente de implementación")


if __name__ == "__main__":
    run_evaluation()
