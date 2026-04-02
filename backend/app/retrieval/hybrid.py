"""
Hybrid Retriever — Combinación de búsqueda semántica y léxica.

Responsabilidad:
- Combinar resultados de ChromaDB (semántico) y BM25 (léxico)
- Usar Reciprocal Rank Fusion (RRF) para fusionar rankings
- Ponderar la contribución de cada retriever

RRF (Reciprocal Rank Fusion):
- Combina rankings de múltiples fuentes sin necesidad de normalizar scores
- Para cada documento, su score RRF = Σ 1/(k + rank_i) donde k=60 (constante)
- Esto produce un ranking unificado robusto
"""

from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from loguru import logger

from app.config import get_settings


def create_hybrid_retriever(semantic_retriever, bm25_retriever) -> EnsembleRetriever:
    """
    Crea un retriever híbrido que combina semántico + BM25 con RRF.

    Args:
        semantic_retriever: Retriever de ChromaDB (similarity search).
        bm25_retriever: Retriever BM25 (keyword search).

    Returns:
        EnsembleRetriever con pesos configurados desde .env

    TODO: Implementar
    - EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[settings.semantic_weight, settings.bm25_weight]
      )
    """
    settings = get_settings()
    logger.info(
        f"Creando hybrid retriever | "
        f"Semántico: {settings.semantic_weight} | BM25: {settings.bm25_weight}"
    )

    return EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[settings.semantic_weight, settings.bm25_weight]
    )

def retrieve(retriever: EnsembleRetriever, query: str) -> list[Document]:
    """
    Ejecuta la búsqueda híbrida para una consulta.

    Args:
        retriever: EnsembleRetriever configurado.
        query: Pregunta del usuario.

    Returns:
        Lista de Documents fusionados y de-duplicados.

    TODO: Implementar
    - retriever.invoke(query)
    - De-duplicar por contenido (puede haber overlap entre semántico y BM25)
    - Loggear cantidad de resultados
    """
    results = retriever.invoke(query)
    
    # De-duplicación estricta por contenido exacto de página.
    # El EnsembleRetriever de Langchain hace el ranking RRF,
    # pero puede haber solapamientos si falló el macheo exacto de objetos.
    unique_docs = []
    seen_content = set()
    for doc in results:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            unique_docs.append(doc)
            
    logger.info(f"Búsqueda híbrida retornó {len(unique_docs)} documentos únicos para: '{query}'")
    
    return unique_docs
