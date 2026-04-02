"""
Run Ingestion — Orquestador del pipeline de ingesta del corpus Docker.

Ejecuta el pipeline completo:
1. Cargar archivos Markdown del corpus (loader)
2. Preprocesar: limpiar shortcodes, HTML, extraer frontmatter (preprocessor)
3. Dividir en chunks con overlap (chunker)
4. Enriquecer metadata de cada chunk (metadata)
5. Indexar chunks en ChromaDB con embeddings (vector_store) ← TODO: retrieval module

Uso (con el venv activado, desde /backend):
    python -m app.ingestion.run
    python -m app.ingestion.run --force   # Forzar re-indexación

Los embeddings persisten en el volumen Docker de ChromaDB.
Solo es necesario re-ejecutar si se actualiza el corpus.
"""

import sys
import time
from pathlib import Path

from loguru import logger

from app.config import get_settings
from app.ingestion.loader import load_markdown_files
from app.ingestion.preprocessor import preprocess_documents
from app.ingestion.chunker import chunk_documents
from app.ingestion.metadata import enrich_metadata


# Ruta al corpus procesado (relativa al archivo run.py)
CORPUS_DIR = str(
    Path(__file__).resolve().parent.parent.parent.parent / "corpus" / "processed"
)


def run_ingestion(force: bool = False) -> list:
    """
    Ejecuta el pipeline completo de ingesta.

    Args:
        force: Si True, fuerza re-indexación aunque ChromaDB ya tenga datos.

    Returns:
        Lista de chunks procesados y listos para indexar.
    """
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("PIPELINE DE INGESTA")
    logger.info("=" * 60)
    logger.info(f"Corpus: {CORPUS_DIR}")
    logger.info(f"Modelo embeddings: {settings.embedding_model}")
    logger.info(f"Chunk size: {settings.chunk_size} | Overlap: {settings.chunk_overlap}")
    logger.info(f"ChromaDB: {settings.chroma_host}:{settings.chroma_port}")
    logger.info(f"Colección: {settings.chroma_collection_name}")
    logger.info(f"Forzar re-indexación: {force}")
    logger.info("=" * 60)

    start_time = time.perf_counter()

    # ──────────────────────────────────────────────
    # Paso 1: Cargar documentos
    # ──────────────────────────────────────────────
    logger.info("Paso 1/4 — Cargando documentos Markdown...")
    documents = load_markdown_files(CORPUS_DIR)
    t1 = time.perf_counter()
    logger.info(f"  → {len(documents)} documentos cargados ({t1 - start_time:.1f}s)")

    # ──────────────────────────────────────────────
    # Paso 2: Preprocesar
    # ──────────────────────────────────────────────
    logger.info("Paso 2/4 — Preprocesando documentos...")
    documents = preprocess_documents(documents)
    t2 = time.perf_counter()
    logger.info(f"  → {len(documents)} documentos válidos ({t2 - t1:.1f}s)")

    # ──────────────────────────────────────────────
    # Paso 3: Chunking
    # ──────────────────────────────────────────────
    logger.info("Paso 3/4 — Dividiendo en chunks...")
    chunks = chunk_documents(documents)
    t3 = time.perf_counter()
    logger.info(f"  → {len(chunks)} chunks generados ({t3 - t2:.1f}s)")

    # ──────────────────────────────────────────────
    # Paso 4: Enriquecer metadata
    # ──────────────────────────────────────────────
    logger.info("Paso 4/4 — Enriqueciendo metadata...")
    chunks = enrich_metadata(chunks)
    t4 = time.perf_counter()
    logger.info(f"  → Metadata enriquecida ({t4 - t3:.1f}s)")

    # ──────────────────────────────────────────────
    # Paso 5 — Indexar en ChromaDB}
    # ──────────────────────────────────────────────
    logger.info("Paso 5/5 — Indexando en ChromaDB con Ollama embeddings...")
    from app.retrieval.vector_store import index_documents
    indexed_count = index_documents(chunks, force=force)
    
    t5 = time.perf_counter()
    logger.info(f"  → {indexed_count} chunks de embeddings procesados ({t5 - t4:.1f}s)")

    # ──────────────────────────────────────────────
    # Resumen
    # ──────────────────────────────────────────────
    total_time = time.perf_counter() - start_time

    logger.info("=" * 60)
    logger.info("INGESTA COMPLETADA")
    logger.info(f"  Documentos cargados:  {len(documents)}")
    logger.info(f"  Chunks generados:     {len(chunks)}")
    logger.info(f"  Tiempo total:         {total_time:.1f}s")
    logger.info("=" * 60)

    return chunks


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    run_ingestion(force=force_flag)
