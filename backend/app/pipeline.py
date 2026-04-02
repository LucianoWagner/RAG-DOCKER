"""
Pipeline RAG — Orquestador principal del flujo completo.

Este módulo conecta todos los componentes del pipeline:
  Consulta → Retrieval Híbrido → Reranking → Evidence Check → Generación

Es el punto de entrada que el endpoint /query y el pipe de OpenWebUI invocan.
No contiene lógica de negocio propia, solo orquesta los módulos.
"""

from langchain_core.documents import Document
from loguru import logger

from app.config import get_settings
from app.models import RAGResponse, EvidenceVerdict


class RAGPipeline:
    """
    Orquestador del pipeline RAG completo.

    Flujo:
    1. Recibir pregunta del usuario
    2. Ejecutar retrieval híbrido (semántico + BM25)
    3. Reranking de resultados
    4. Verificar evidencia suficiente
    5. Generar respuesta con citas (o abstenerse)
    6. Retornar RAGResponse

    Uso:
        pipeline = RAGPipeline()
        response = pipeline.run("¿Cómo instalo Docker en Windows?")
    """

    def __init__(self):
        """
        Inicializa el pipeline cargando todos los componentes.

        TODO: Implementar
        - Inicializar vector store (retrieval.vector_store)
        - Inicializar BM25 retriever (retrieval.bm25_retriever)
        - Crear hybrid retriever (retrieval.hybrid)
        - Inicializar LLM (generation.generator)
        - Cargar chunks de ChromaDB para BM25 (o desde archivo serializado)

        Nota: El BM25 necesita los chunks en memoria. Se deben cargar
        desde ChromaDB al iniciar el pipeline, no re-indexar.
        """
        self.settings = get_settings()
        logger.info("Inicializando RAG Pipeline...")

        # TODO: Inicializar componentes
        # self.vector_store = ...
        # self.bm25_retriever = ...
        # self.hybrid_retriever = ...
        # self.llm = ...

        logger.info("RAG Pipeline inicializado")

    def run(self, question: str) -> RAGResponse:
        """
        Ejecuta el pipeline RAG completo para una pregunta.

        Args:
            question: Pregunta del usuario en lenguaje natural.

        Returns:
            RAGResponse con respuesta, citas, evidencia y metadata.

        TODO: Implementar la orquestación:

        1. RETRIEVAL HÍBRIDO
           retrieved_chunks = hybrid.retrieve(self.hybrid_retriever, question)

        2. RERANKING
           reranked_chunks = reranker.rerank_documents(question, retrieved_chunks)

        3. EVIDENCE CHECK
           evidence = evidence_checker.check_evidence(question, reranked_chunks)
           if evidence.verdict == EvidenceVerdict.INSUFFICIENT:
               return abstention response

        4. GENERACIÓN
           context = prompt_templates.format_context(reranked_chunks)
           messages = prompt_templates.build_messages(question, context)
           response = generator.generate_response(question, reranked_chunks, evidence, messages)

        5. RETURN
           return response
        """
        logger.info(f"Pipeline RAG ejecutando | Pregunta: {question[:80]}...")

        # TODO: Implementar orquestación

        # Placeholder
        from app.models import EvidenceResult
        return RAGResponse(
            answer="🚧 Pipeline RAG en construcción.",
            sources=[],
            evidence=EvidenceResult(
                verdict=EvidenceVerdict.INSUFFICIENT,
                top_score=0.0,
                relevant_count=0,
                details="Pipeline no implementado aún",
            ),
            retrieval_metadata={"question": question},
        )
