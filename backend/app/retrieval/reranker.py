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

from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
from loguru import logger

from app.config import get_settings

# Inicializamos el ranker globally para evitar sobrecarga en cada request
# Descargará un pequeño modelo cross-encoder (ms-marco-TinyBERT o similar)
# la primera vez que se ejecute en local.
try:
    logger.info("Cargando modelo FlashRank (reranker)...")
    _ranker = Ranker()
except Exception as e:
    logger.warning(f"No se pudo cargar FlashRank prematuramente: {e}")
    _ranker = None

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

    if not documents:
        return []

    global _ranker
    if _ranker is None:
        _ranker = Ranker()

    # Formatear el input para FlashRank: lista de diccionarios
    passages = []
    for i, doc in enumerate(documents):
        passages.append({
            "id": i,
            "text": doc.page_content,
            "meta": doc.metadata
        })

    # Crear request de reranking
    rerank_req = RerankRequest(query=query, passages=passages)
    
    # Ejecutar cross-encoder
    # Esto le asignará un 'score' a cada uno y retornará la lista reordenada
    results = _ranker.rerank(rerank_req)

    # Seleccionar top-N y reconstruir los objetos Document de LangChain
    top_n = settings.rerank_top_n
    ranked_docs = []
    
    for rank, res in enumerate(results[:top_n]):
        doc = Document(page_content=res["text"], metadata=res.get("meta", {}))
        # Agregamos data valiosa a la metadata
        doc.metadata["rerank_score"] = res.get("score", 0.0)
        doc.metadata["rerank_position"] = rank + 1
        ranked_docs.append(doc)

    logger.info(f"  → Reranking completado, seleccionados los top {len(ranked_docs)}")
    return ranked_docs
