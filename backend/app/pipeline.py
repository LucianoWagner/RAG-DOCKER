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

        from app.retrieval.vector_store import get_vector_store, get_semantic_retriever
        from app.retrieval.bm25_retriever import create_bm25_retriever
        from app.retrieval.hybrid import create_hybrid_retriever
        from app.generation.generator import get_llm

        # 1. Inicialización de clientes core
        self.vector_store = get_vector_store()
        self.semantic_retriever = get_semantic_retriever()
        
        # 2. Re-construcción del BM25 Index desde ChromaDB
        res = self.vector_store._collection.get(include=["documents", "metadatas"])
        chunks = []
        if res and res["documents"]:
            for text, meta in zip(res["documents"], res["metadatas"]):
                chunks.append(Document(page_content=text, metadata=meta))
                
        if chunks:
            self.bm25_retriever = create_bm25_retriever(chunks)
            # 3. Empaquetado Híbrido
            self.hybrid_retriever = create_hybrid_retriever(self.semantic_retriever, self.bm25_retriever)
        else:
            logger.warning("Pipeline no encontró chunks. Usa BM25 = None.")
            self.bm25_retriever = None
            self.hybrid_retriever = self.semantic_retriever  # fallback
            
        # 4. Motor Generativo
        self.llm = get_llm()

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

        from app.retrieval.hybrid import retrieve
        from app.retrieval.reranker import rerank_documents
        from app.generation.evidence_checker import check_evidence, get_abstention_response
        from app.generation.prompt_templates import format_context, build_messages
        from app.generation.generator import generate_response

        # 1. RETRIEVAL HÍBRIDO
        retrieved_chunks = retrieve(self.hybrid_retriever, question) if self.bm25_retriever else self.hybrid_retriever.invoke(question)

        # 2. RERANKING
        reranked_chunks = rerank_documents(question, retrieved_chunks)

        # 3. EVIDENCE CHECK
        evidence = check_evidence(question, reranked_chunks)
        if evidence.verdict == EvidenceVerdict.INSUFFICIENT:
            logger.warning("Evidencia insuficiente. Resolviendo con abstención programada.")
            return RAGResponse(
                answer=get_abstention_response(),
                sources=[],
                evidence=evidence,
                retrieval_metadata={
                    "question": question,
                    "chunks_used": 0,
                    "status": "abstained"
                }
            )

        # 4. GENERACIÓN
        logger.info("Evidencia validada. Formateando contexto y delegando a Ollama...")
        context = format_context(reranked_chunks)
        messages = build_messages(question, context)
        response = generate_response(question, reranked_chunks, evidence, messages)

        # 5. RETURN
        return response
