"""
Reranker — Re-evaluación y re-ordenamiento de chunks recuperados.

Responsabilidad:
- Tomar los top-K chunks del retriever híbrido
- Re-evaluar la relevancia de cada chunk respecto a la query
  usando un modelo cross-encoder
- Retornar los top-N chunks más relevantes

¿Por qué reranking?
- Los retrievers iniciales (semántico + BM25) usan representaciones
  "bi-encoder": query y documento se codifican por separado.
- Un reranker usa un "cross-encoder": analiza la relación query-documento
  de forma conjunta, lo que es más preciso pero más costoso.
- Se aplica solo sobre los ~15-20 resultados del retriever, no sobre todo el corpus.

Herramienta: FlashRank (ultra-ligero, local, sin API externa).
"""

from langchain_core.documents import Document
from loguru import logger

from app.config import get_settings


def rerank_documents(query: str, documents: list[Document]) -> list[Document]:
    """
    Re-rankea documentos por relevancia usando FlashRank.

    Args:
        query: Pregunta del usuario.
        documents: Chunks recuperados por el hybrid retriever.

    Returns:
        Lista de los top-N chunks re-rankeados, ordenados por relevancia.
        Cada documento tendrá metadata adicional:
        - rerank_score: score del cross-encoder
        - original_rank: posición antes del reranking

    TODO: Implementar
    - Inicializar FlashRank (flashrank.Ranker)
    - Formatear input como lista de {"query": ..., "text": ...}
    - Ejecutar reranking
    - Seleccionar top-N (settings.rerank_top_n)
    - Agregar rerank_score a metadata de cada chunk
    - Retornar ordenados por score descendente
    """
    settings = get_settings()
    logger.info(
        f"Reranking {len(documents)} documentos | Top-N: {settings.rerank_top_n}"
    )

    # TODO: Implementar
    return documents[:settings.rerank_top_n]
