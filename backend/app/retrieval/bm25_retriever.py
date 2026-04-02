"""
BM25 Retriever — Búsqueda léxica por keywords.

Responsabilidad:
- Mantener un índice BM25 en memoria sobre los chunks del corpus
- Complementar la búsqueda semántica para capturar keywords exactos
  (nombres de comandos, mensajes de error, flags, etc.)

Ventajas de BM25 en soporte técnico de Docker:
- Captura exacta de comandos: "docker compose up", "docker ps"
- Captura exacta de errores: "Cannot connect to the Docker daemon"
- No depende de la calidad de los embeddings para matches literales

Nota: BM25Retriever de LangChain carga todo en memoria.
Con el corpus acotado (~cientos de chunks) no debería ser problema.
"""

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from loguru import logger

from app.config import get_settings


def create_bm25_retriever(chunks: list[Document]) -> BM25Retriever:
    """
    Crea un retriever BM25 a partir de los chunks del corpus.

    Args:
        chunks: Lista de chunks indexados (los mismos de ChromaDB).

    Returns:
        BM25Retriever configurado con top_k del .env

    TODO: Implementar
    - Usar BM25Retriever.from_documents(chunks)
    - Configurar k = settings.top_k
    - Retornar el retriever

    Nota: Este retriever se debe crear al iniciar el backend
    y mantenerse en memoria. Se reconstruye si se re-ingesta el corpus.
    """
    settings = get_settings()
    logger.info(f"Creando BM25 retriever con {len(chunks)} chunks, k={settings.top_k}")

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = settings.top_k
    return bm25_retriever
