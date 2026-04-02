"""
Configuración centralizada del proyecto RAG.

Todas las variables de entorno se cargan aquí y se exponen como un objeto
Settings que se puede inyectar en cualquier módulo via get_settings().
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración del backend RAG, cargada desde variables de entorno."""

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"

    # --- ChromaDB ---
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "docker_docs"

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 200

    # --- Retrieval ---
    top_k: int = 10
    rerank_top_n: int = 5
    semantic_weight: float = 0.6
    bm25_weight: float = 0.4

    # --- Evidence Check ---
    min_top_score: float = 0.3
    min_relevant_chunks: int = 2
    relevance_threshold: float = 0.25

    # --- Logging ---
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Singleton de configuración. Se cachea para evitar re-lectura."""
    return Settings()
