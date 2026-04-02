"""
Vector Store — Wrapper sobre ChromaDB para indexación y búsqueda vectorial.

Responsabilidad:
- Conectar con ChromaDB (servicio Docker)
- Crear/obtener la collection para el corpus Docker
- Generar embeddings con Ollama
- Indexar chunks resultantes de la ingesta
- Crear y devolver el retriever semántico

Notas:
- ChromaDB persiste en el volumen Docker de la máquina host.
- Solo se re-indexa si se especifica que se quiere forzar la reindexación o 
  la colección está vacía.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from loguru import logger

from app.config import get_settings


def get_embedding_function() -> OllamaEmbeddings:
    """
    Crea la función de embeddings usando Ollama.

    Returns:
        OllamaEmbeddings configurado con el modelo y URL del .env.
    """
    settings = get_settings()
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def get_chroma_client() -> chromadb.HttpClient:
    """Obtiene un cliente puro de ChromaDB conectado al server Docker."""
    settings = get_settings()
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        settings=ChromaSettings(allow_reset=True),
    )


def get_vector_store() -> Chroma:
    """
    Obtiene la instancia de LangChain ChromaDB.

    Returns:
        Chroma vector store conectado al servicio ChromaDB Docker,
        usando Ollama para embeddings de forma automática.
    """
    settings = get_settings()
    return Chroma(
        client=get_chroma_client(),
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embedding_function(),
    )


def index_documents(chunks: list[Document], force: bool = False) -> int:
    """
    Indexa fragmentos (chunks) en ChromaDB generando sus embeddings.

    Args:
        chunks: Lista de chunks con metadata.
        force: Si True, borra todo lo viejo y re-indexa desde cero.

    Returns:
        Cantidad total de fragmentos insertados.
    """
    settings = get_settings()
    logger.info(f"Conectando a ChromaDB en {settings.chroma_host}:{settings.chroma_port}...")
    
    vector_store = get_vector_store()
    
    # Verificamos si ya hay documentos indexados. 
    # El método para obtener count varía entre cliente nativo y LangChain,
    # usamos el nativo para más seguridad.
    client = get_chroma_client()
    try:
        col = client.get_collection(settings.chroma_collection_name)
        count = col.count()
    except Exception:
        count = 0

    if count > 0 and not force:
        logger.info(f"ChromaDB ya tiene {count} chunks. Omitiendo indexación.")
        logger.info("Usá --force en la ejecución de ingesta para re-indexar.")
        return 0

    if force and count > 0:
        logger.warning(f"Borrando colección existente con {count} documentos (--force)...")
        # Borrar y recrear todo
        client.delete_collection(settings.chroma_collection_name)
        # Volvemos a instanciar VectorStore para que cree la colección de nuevo
        vector_store = get_vector_store()

    if not chunks:
        logger.warning("No hay chunks para indexar.")
        return 0

    logger.info(f"Indexando {len(chunks)} chunks con embeddings de Ollama...")
    logger.info("⏳ Esto tardará un par de minutos dependiendo de tu máquina.")
    
    # add_documents genera automáticamente los embeddings usando el embedding_function 
    # (nomic-embed-text) y los sube a ChromaDB
    vector_store.add_documents(chunks)
    
    logger.info("✅ Embeddings generados e indexados correctamente.")
    return len(chunks)


def get_semantic_retriever() -> VectorStoreRetriever:
    """
    Crea un retriever semántico basado en ChromaDB.

    Returns:
        Un VectorStoreRetriever configurado con el top_k de k vecinos.
    """
    settings = get_settings()
    vector_store = get_vector_store()
    
    return vector_store.as_retriever(
        search_kwargs={"k": settings.top_k}
    )
