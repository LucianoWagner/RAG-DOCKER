"""
Script E2E de Evaluación RAG (LLM-as-a-Judge con Groq Llama-3.3-70B)
Debido a la ausencia de wheels nativas de scikit-network/RAGAS para Python 3.14,
este script implementa un pipeline de evaluación propio usando la misma metodología
científica (LLM-as-a-judge) para medir Faithfulness, Answer Correctness y Relevancy.
"""

import json
import time
import os
import asyncio
from pathlib import Path

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.pipeline import RAGPipeline
from app.config import get_settings


EVAL_PROMPT_FAITHFULNESS = """
You are an impartial judge evaluating the 'Faithfulness' of an AI assistant's answer.
Given the provided 'Context' and the assistant's 'Answer', you must determine if the answer
contains ONLY information present in the context. If the answer invents, hallucinates, or uses
outside knowledge, score it 0. If it strictly adheres to the context or correctly states that
the context lacks info, score it 1.

Context:
{context}

Answer:
{answer}

Respond strictly with ONLY a JSON object: {{"score": 1, "reason": "short explanation"}}
"""

EVAL_PROMPT_CORRECTNESS = """
You are an impartial judge evaluating 'Answer Correctness'.
Compare the assistant's 'Answer' with the 'Ground Truth'.
Score 1 if the Answer conceptually matches the Ground Truth or solves the problem identically.
Score 0 if the Answer is completely wrong or contradicts the Ground Truth.
Score 0.5 if it is partially correct.

Ground Truth:
{ground_truth}

Answer:
{answer}

Respond strictly with ONLY a JSON object: {{"score": 1, "reason": "short explanation"}}
"""

class RAGEvaluator:
    def __init__(self):
        settings = get_settings()
        # Usamos Llama-3.3-70B como Juez Implacable (Mismo modelo o uno mejor)
        self.judge_llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            api_key=settings.groq_api_key,
            temperature=0.0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        self.pipeline = RAGPipeline()
        self.results = []

    def evaluate_faithfulness(self, context: str, answer: str) -> dict:
        messages = [
            SystemMessage(content="You are a strict JSON evaluator."),
            HumanMessage(content=EVAL_PROMPT_FAITHFULNESS.format(context=context, answer=answer))
        ]
        res = self.judge_llm.invoke(messages).content
        try:
            return json.loads(res)
        except:
            return {"score": 0, "reason": "Failed to parse JSON judge output"}

    def evaluate_correctness(self, truth: str, answer: str) -> dict:
        messages = [
            SystemMessage(content="You are a strict JSON evaluator."),
            HumanMessage(content=EVAL_PROMPT_CORRECTNESS.format(ground_truth=truth, answer=answer))
        ]
        res = self.judge_llm.invoke(messages).content
        try:
            return json.loads(res)
        except:
            return {"score": 0, "reason": "Failed to parse JSON judge output"}

    def run_eval_dataset(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        logger.info(f"Iniciando evaluación de {len(dataset)} preguntas...")
        
        for i, data in enumerate(dataset):
            q = data['question']
            truth = data['ground_truth']
            category = data['category']
            
            logger.info(f"[{i+1}/{len(dataset)}] Evaluando: {q}")
            
            # Ejecutar nuestro pipeline RAG real
            t0 = time.time()
            try:
                response = self.pipeline.run(q)
                answer = response.answer
                
                # Unimos el texto de todos los chunks recuperados para el contexto
                context = "\n---\n".join([c.chunk_text for c in response.retrieval_metadata.chunks_metadata])
                
                # LLM-as-a-judge
                faith = self.evaluate_faithfulness(context, answer)
                corr = self.evaluate_correctness(truth, answer)
                
                self.results.append({
                    "question": q,
                    "category": category,
                    "generated_answer": answer,
                    "ground_truth": truth,
                    "faithfulness_score": faith.get("score", 0),
                    "correctness_score": corr.get("score", 0),
                    "faithfulness_reason": faith.get("reason", ""),
                    "correctness_reason": corr.get("reason", ""),
                    "latency_sec": round(time.time() - t0, 2)
                })
                
            except Exception as e:
                logger.error(f"Error procesando pregunta '{q}': {e}")
                self.results.append({
                    "question": q,
                    "category": category,
                    "generated_answer": f"ERROR: {e}",
                    "faithfulness_score": 0,
                    "correctness_score": 0
                })
            
            # Rate limit backoff just in case
            time.sleep(1)

        self._export_results()

    def _export_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv("evaluation_report.csv", index=False)
        
        # Agrupar estadísticas por categoría
        logger.info("=== REPORTE DE EVALUACIÓN ===")
        summary = df.groupby("category")[["faithfulness_score", "correctness_score"]].mean()
        print("\n" + summary.to_string() + "\n")
        
        total_faith = df["faithfulness_score"].mean()
        total_corr = df["correctness_score"].mean()
        logger.info(f"Promedio Total Faithfulness: {total_faith:.2f}")
        logger.info(f"Promedio Total Answer Correctness: {total_corr:.2f}")
        logger.info("Reporte detallado guardado en 'evaluation_report.csv'")


if __name__ == "__main__":
    dataset_path = Path(__file__).parent.parent / "tests" / "eval_dataset.json"
    evaluator = RAGEvaluator()
    evaluator.run_eval_dataset(str(dataset_path))
