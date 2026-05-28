"""
test_evaluation.py — Evaluación E2E del RAG con RAGAS nativo.

Ejecuta el pipeline RAG sobre el dataset de prueba y calcula métricas
reales con RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall).

Uso:
    # Desde backend/ con venv-eval activado:
    python -m tests.test_evaluation --model llama-3.3-70b-versatile
    python -m tests.test_evaluation --model llama-3.1-8b-instant
    python -m tests.test_evaluation --model llama-3.3-70b-versatile --delay 15
    python -m tests.test_evaluation --quick   # Solo 10 preguntas (2 por categoría)
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from datasets import Dataset
from loguru import logger

# ── RAGAS imports ─────────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from openai import OpenAI

# ── LangChain imports ─────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings

# ── Pipeline local ────────────────────────────────────────────────────
from app.config import get_settings
from app.pipeline import RAGPipeline


# =====================================================================
# Configuración
# =====================================================================

EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
RESULTS_DIR = Path(__file__).parent
COLLECTED_DIR = RESULTS_DIR / "eval_runs"
DEFAULT_DELAY = 15  # segundos entre preguntas para respetar rate limits

# Categorías para --quick (2 preguntas por categoría = 10 total)
QUICK_CATEGORIES = {
    "installation": 2,
    "getting_started": 2,
    "cli_reference": 2,
    "troubleshooting": 2,
    "configuration": 2,
}


def load_dataset(path: str | Path, quick: bool = False) -> list[dict]:
    """Carga el dataset de evaluación desde JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if quick:
        # Tomar solo las primeras N preguntas por categoría
        filtered = []
        counts = {}
        for item in data:
            cat = item["category"]
            max_per_cat = QUICK_CATEGORIES.get(cat, 2)
            counts[cat] = counts.get(cat, 0) + 1
            if counts[cat] <= max_per_cat:
                filtered.append(item)
        logger.info(f"Modo --quick: {len(filtered)} preguntas seleccionadas")
        return filtered

    return data


def get_ragas_llm(provider: str = "ollama", model_name: str | None = None):
    """
    Crea el LLM juez para RAGAS usando Ollama local vía API OpenAI-compatible.
    Usa llm_factory (requerido por RAGAS v0.4.x metrics).
    """
    settings = get_settings()

    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY no encontrada. Agregala a tu archivo .env")
        judge_model = model_name or "llama-3.1-8b-instant"
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key,
        )
        logger.info(f"Juez RAGAS: Groq ({judge_model})")
    else:
        judge_model = model_name or settings.llm_model
        client = OpenAI(
            base_url=f"{settings.ollama_base_url}/v1",
            api_key="ollama",
        )
        logger.info(f"Juez RAGAS: Ollama local ({judge_model} en {settings.ollama_base_url})")

    llm = llm_factory(judge_model, client=client)
    return llm


def get_ragas_embeddings():
    """
    Crea los embeddings para RAGAS usando Ollama local.
    Usa LangchainEmbeddingsWrapper (AnswerRelevancy necesita embed_query).
    """
    settings = get_settings()
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )
    return LangchainEmbeddingsWrapper(embeddings)


# =====================================================================
# Pipeline de Evaluación
# =====================================================================

def run_pipeline_on_dataset(
    pipeline: RAGPipeline,
    dataset: list[dict],
    delay: int = DEFAULT_DELAY,
) -> dict:
    """
    Ejecuta el RAGPipeline sobre cada pregunta del dataset y recopila
    los datos que RAGAS necesita para evaluar.

    Returns:
        Dict con listas: question, answer, contexts, ground_truth, category, latency
    """
    results = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "category": [],
        "latency": [],
    }

    total = len(dataset)
    for i, item in enumerate(dataset):
        q = item["question"]
        logger.info(f"[{i+1}/{total}] Procesando: {q[:70]}...")

        t0 = time.time()
        try:
            response = pipeline.run(q)
            answer = response.answer
            # Extraer textos crudos de los chunks recuperados
            contexts = [
                c.chunk_text for c in response.retrieval_metadata.chunks_metadata
            ]
            # Filtrar contextos vacíos
            contexts = [c for c in contexts if c.strip()]
            if not contexts:
                contexts = ["No context retrieved."]
        except Exception as e:
            logger.error(f"Error en pregunta '{q[:50]}': {e}")
            answer = f"ERROR: {e}"
            contexts = ["No context retrieved due to error."]

        latency = round(time.time() - t0, 2)

        results["question"].append(q)
        results["answer"].append(answer)
        results["contexts"].append(contexts)
        results["ground_truth"].append(item["ground_truth"])
        results["category"].append(item["category"])
        results["latency"].append(latency)

        logger.info(f"  ✓ Respondida en {latency}s | Chunks: {len(contexts)}")

        # Rate-limit delay (no aplica al último)
        if i < total - 1:
            logger.info(f"  ⏳ Esperando {delay}s (rate-limit)...")
            time.sleep(delay)

    return results


def save_collected_data(
    data: dict,
    model_name: str,
    output_path: str | Path | None = None,
    quick: bool = False,
) -> Path:
    """Guarda respuestas y contextos ya generados para evaluar RAGAS luego."""
    COLLECTED_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        safe_model = model_name.replace(".", "_").replace("-", "_")
        mode = "quick" if quick else "full"
        output_path = COLLECTED_DIR / f"collected_{safe_model}_{mode}_{timestamp}.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "generator_model": model_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "items": len(data.get("question", [])),
            "quick": quick,
        },
        "data": data,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"Datos de evaluacion guardados en: {output_path}")
    return output_path


def load_collected_data(input_path: str | Path) -> dict:
    """Carga un JSON generado por --phase collect."""
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "data" in payload:
        logger.info(f"Cargando datos pre-generados: {input_path}")
        logger.info(f"Metadata: {payload.get('metadata', {})}")
        return payload["data"]

    return payload


def _evaluate_ragas_dataset(
    ragas_dataset: Dataset,
    metrics: list,
    run_config: RunConfig,
) -> pd.DataFrame:
    result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        run_config=run_config,
    )
    return result.to_pandas()


def run_ragas_evaluation(
    data: dict,
    judge_provider: str = "ollama",
    judge_model: str | None = None,
    ragas_row_delay: int = 0,
    max_retries: int = 5,
    max_wait: int = 60,
    timeout: int = 600,
) -> pd.DataFrame:
    """
    Ejecuta la evaluación RAGAS con los datos recopilados del pipeline.

    El juez es siempre Ollama local (sin rate-limits ni costos).
    Usa RunConfig(max_workers=1) para serializar llamadas.

    Args:
        data: Dict con question, answer, contexts, ground_truth

    Returns:
        DataFrame con scores por pregunta
    """
    logger.info("Iniciando evaluación RAGAS con juez LOCAL (Ollama)")

    # Preparar dataset RAGAS
    ragas_data = {
        "question": data["question"],
        "answer": data["answer"],
        "contexts": data["contexts"],
        "ground_truth": data["ground_truth"],
    }
    ragas_dataset = Dataset.from_dict(ragas_data)

    # Configurar juez local (Ollama)
    evaluator_llm = get_ragas_llm()
    evaluator_embeddings = get_ragas_embeddings()

    # Instanciar métricas con el LLM juez (RAGAS v0.4.x requiere objetos instanciados)
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    # RunConfig para Ollama local:
    # - max_workers=1: Serializado (Ollama procesa de a uno)
    # - timeout=600: 10 min por operación (modelos locales son más lentos)
    run_config = RunConfig(
        max_workers=1,
        max_retries=5,
        max_wait=60,
        timeout=600,
    )

    # Ejecutar evaluación
    result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        run_config=run_config,
    )

    # Convertir a DataFrame y agregar metadata
    df = result.to_pandas()
    df["category"] = data["category"]
    df["latency"] = data["latency"]

    return df


def run_ragas_evaluation(
    data: dict,
    judge_provider: str = "ollama",
    judge_model: str | None = None,
    ragas_row_delay: int = 0,
    max_retries: int = 5,
    max_wait: int = 60,
    timeout: int = 600,
) -> pd.DataFrame:
    """Ejecuta RAGAS con juez local o Groq. Esta definicion pisa la legacy."""
    logger.info(f"Iniciando evaluacion RAGAS | juez={judge_provider} | modelo={judge_model or 'default'}")

    ragas_data = {
        "question": data["question"],
        "answer": data["answer"],
        "contexts": data["contexts"],
        "ground_truth": data["ground_truth"],
    }
    ragas_dataset = Dataset.from_dict(ragas_data)

    evaluator_llm = get_ragas_llm(judge_provider, judge_model)
    evaluator_embeddings = get_ragas_embeddings()

    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    run_config = RunConfig(
        max_workers=1,
        max_retries=max_retries,
        max_wait=max_wait,
        timeout=timeout,
    )

    if ragas_row_delay > 0:
        logger.info(f"Evaluando RAGAS fila por fila con pausa de {ragas_row_delay}s")
        frames = []
        total = len(data["question"])
        for i in range(total):
            logger.info(f"RAGAS [{i + 1}/{total}]")
            row_dataset = ragas_dataset.select([i])
            row_df = _evaluate_ragas_dataset(row_dataset, metrics, run_config)
            row_df["category"] = [data["category"][i]]
            row_df["latency"] = [data["latency"][i]]
            frames.append(row_df)

            if i < total - 1:
                logger.info(f"Esperando {ragas_row_delay}s antes de la siguiente fila RAGAS...")
                time.sleep(ragas_row_delay)

        return pd.concat(frames, ignore_index=True)

    df = _evaluate_ragas_dataset(ragas_dataset, metrics, run_config)
    df["category"] = data["category"]
    df["latency"] = data["latency"]
    return df


def print_report(df: pd.DataFrame, model_name: str, output_path: Path):
    """Imprime reporte formateado y exporta CSV."""
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print("\n" + "=" * 70)
    print(f"  REPORTE RAGAS — Modelo: {model_name}")
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Resumen por categoría
    print("\n📊 Promedios por Categoría:\n")
    summary = df.groupby("category")[metric_cols].mean()
    print(summary.to_string(float_format="%.3f"))

    # Promedio general
    print("\n" + "-" * 70)
    print("📈 Promedio General:\n")
    for col in metric_cols:
        avg = df[col].mean()
        emoji = "✅" if avg >= 0.7 else "⚠️" if avg >= 0.5 else "❌"
        print(f"  {emoji} {col:25s}: {avg:.3f}")

    avg_latency = df["latency"].mean()
    print(f"\n  ⏱️  {'Latencia promedio':25s}: {avg_latency:.2f}s")
    print("=" * 70)

    # Exportar CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_model = model_name.replace(".", "_").replace("-", "_")
    csv_path = output_path / f"ragas_report_{safe_model}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Reporte CSV guardado en: {csv_path}")

    return summary


# =====================================================================
# Entrypoint
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluación RAGAS del pipeline RAG")
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.3-70b-versatile",
        help="Modelo Groq a usar como generador (default: llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=DEFAULT_DELAY,
        help=f"Segundos de espera entre preguntas para rate-limit (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Ejecutar solo 10 preguntas (2 por categoría) para testeo rápido",
    )
    args = parser.parse_args()

    logger.info(f"Modelo generador: {args.model}")
    logger.info(f"Juez RAGAS: Ollama local (sin rate-limits)")
    logger.info(f"Delay entre preguntas: {args.delay}s")
    logger.info(f"Modo quick: {args.quick}")

    # 1. Cargar dataset
    dataset = load_dataset(EVAL_DATASET_PATH, quick=args.quick)
    logger.info(f"Dataset cargado: {len(dataset)} preguntas")

    # 2. Parchear el modelo del generador antes de inicializar el pipeline
    #    Esto permite testear 70b vs 8b sin tocar el código de producción.
    import app.generation.generator as gen_module
    original_get_llm = gen_module.get_llm

    def patched_get_llm():
        settings = get_settings()
        return ChatGroq(
            model_name=args.model,
            api_key=settings.groq_api_key,
            temperature=0.0,
        )

    gen_module.get_llm = patched_get_llm

    # 3. Inicializar pipeline
    logger.info("Inicializando RAG Pipeline...")
    pipeline = RAGPipeline()

    # 4. Correr pipeline sobre dataset
    logger.info("Ejecutando pipeline sobre dataset de evaluación...")
    data = run_pipeline_on_dataset(pipeline, dataset, delay=args.delay)

    # 5. Restaurar función original
    gen_module.get_llm = original_get_llm

    # 6. Evaluar con RAGAS
    logger.info("=" * 50)
    logger.info("FASE 2: Evaluación RAGAS (serializada, max_workers=1)")
    logger.info("Esto puede tardar varios minutos. El juez evalúa cada pregunta de a una.")
    logger.info("=" * 50)
    df = run_ragas_evaluation(data)

    # 7. Reporte
    print_report(df, args.model, RESULTS_DIR)


def main():
    parser = argparse.ArgumentParser(description="Evaluacion RAGAS del pipeline RAG")
    parser.add_argument(
        "--phase",
        choices=["all", "collect", "evaluate"],
        default="all",
        help="all: genera y evalua; collect: solo guarda respuestas/contextos; evaluate: evalua un JSON guardado",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.3-70b-versatile",
        help="Modelo Groq a usar como generador",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=DEFAULT_DELAY,
        help=f"Segundos de espera entre preguntas del pipeline (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Ejecutar solo 10 preguntas (2 por categoria)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Ruta donde guardar el JSON de --phase collect",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="JSON generado por --phase collect para evaluar luego",
    )
    parser.add_argument(
        "--judge-provider",
        choices=["ollama", "groq"],
        default="ollama",
        help="Proveedor del juez RAGAS",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Modelo del juez RAGAS. Para Groq, sugerido: llama-3.1-8b-instant",
    )
    parser.add_argument(
        "--ragas-row-delay",
        type=int,
        default=0,
        help="Si >0, evalua RAGAS una pregunta por vez y espera N segundos entre filas",
    )
    parser.add_argument(
        "--ragas-max-wait",
        type=int,
        default=60,
        help="Espera maxima entre reintentos de RAGAS",
    )
    parser.add_argument(
        "--ragas-timeout",
        type=int,
        default=600,
        help="Timeout por operacion de RAGAS",
    )
    args = parser.parse_args()

    logger.info(f"Fase: {args.phase}")
    logger.info(f"Modelo generador: {args.model}")
    logger.info(f"Juez RAGAS: {args.judge_provider} ({args.judge_model or 'default'})")
    logger.info(f"Delay entre preguntas: {args.delay}s")
    logger.info(f"Modo quick: {args.quick}")

    if args.phase == "evaluate":
        if not args.input_json:
            raise ValueError("--phase evaluate requiere --input-json")
        data = load_collected_data(args.input_json)
        report_model_name = f"{args.model}_judge_{args.judge_provider}_{args.judge_model or 'default'}"
    else:
        dataset = load_dataset(EVAL_DATASET_PATH, quick=args.quick)
        logger.info(f"Dataset cargado: {len(dataset)} preguntas")

        import app.generation.generator as gen_module
        original_get_llm = gen_module.get_llm

        def patched_get_llm():
            settings = get_settings()
            return ChatGroq(
                model_name=args.model,
                api_key=settings.groq_api_key,
                temperature=0.0,
            )

        gen_module.get_llm = patched_get_llm
        try:
            logger.info("Inicializando RAG Pipeline...")
            pipeline = RAGPipeline()
            logger.info("Ejecutando pipeline sobre dataset de evaluacion...")
            data = run_pipeline_on_dataset(pipeline, dataset, delay=args.delay)
        finally:
            gen_module.get_llm = original_get_llm

        collected_path = save_collected_data(
            data,
            args.model,
            output_path=args.output_json,
            quick=args.quick,
        )
        logger.info(f"JSON disponible para evaluar luego: {collected_path}")
        report_model_name = args.model

        if args.phase == "collect":
            return

    logger.info("=" * 50)
    logger.info("FASE 2: Evaluacion RAGAS")
    logger.info("=" * 50)
    df = run_ragas_evaluation(
        data,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        ragas_row_delay=args.ragas_row_delay,
        max_wait=args.ragas_max_wait,
        timeout=args.ragas_timeout,
    )

    print_report(df, report_model_name, RESULTS_DIR)


if __name__ == "__main__":
    main()
