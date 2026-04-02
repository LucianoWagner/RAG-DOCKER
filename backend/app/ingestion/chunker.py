"""
Chunker — División de documentos en fragmentos para indexación.

Responsabilidad:
- Dividir documentos limpios en chunks de tamaño controlado
- Usar separadores Markdown-aware (encabezados → párrafos → líneas)
- Preservar bloques de código como unidades completas cuando sea posible
- Mantener overlap entre chunks para preservar contexto entre fragmentos
- Indexar cada chunk con su posición dentro del documento original

Estrategia: RecursiveCharacterTextSplitter de LangChain con separadores
personalizados que respetan la jerarquía de Markdown.
"""

from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.config import get_settings


# Separadores ordenados por prioridad (de más grueso a más fino).
# El splitter intenta partir por el primer separador. Si un chunk sigue
# siendo muy grande, prueba con el siguiente, y así sucesivamente.
MARKDOWN_SEPARATORS = [
    "\n## ",       # H2 — secciones principales
    "\n### ",      # H3 — subsecciones
    "\n#### ",     # H4 — sub-subsecciones
    "\n```",       # Inicio/fin de bloques de código
    "\n\n",        # Párrafos (doble salto de línea)
    "\n",          # Líneas individuales
    ". ",          # Oraciones (punto + espacio)
    " ",           # Palabras (último recurso)
]


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Divide los documentos en chunks con overlap.

    Args:
        documents: Lista de documentos preprocesados.
        chunk_size: Tamaño de chunk en caracteres. Si es None, usa config.
        chunk_overlap: Solapamiento entre chunks. Si es None, usa config.

    Returns:
        Lista de chunks con metadata heredada + chunk_index y total_chunks.
    """
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    logger.info(f"Chunking {len(documents)} documentos | size={size}, overlap={overlap}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=MARKDOWN_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True,  # Preserva los encabezados ## en el chunk
    )

    # Splitear todos los documentos
    all_chunks = splitter.split_documents(documents)

    # Agregar chunk_index y total_chunks agrupados por documento fuente
    all_chunks = _add_chunk_indices(all_chunks)

    # Estadísticas
    _log_stats(all_chunks, size)

    return all_chunks


def _add_chunk_indices(chunks: list[Document]) -> list[Document]:
    """
    Agrega chunk_index y total_chunks a la metadata de cada chunk,
    agrupados por documento fuente.

    Esto permite saber la posición de un chunk dentro de su documento
    original, útil para trazabilidad y para entender el contexto.
    """
    # Agrupar chunks por source_file
    groups: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")
        groups[source].append(chunk)

    # Asignar índices dentro de cada grupo
    for source, group_chunks in groups.items():
        total = len(group_chunks)
        for idx, chunk in enumerate(group_chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = total

    return chunks


def _log_stats(chunks: list[Document], target_size: int) -> None:
    """Loggea estadísticas de los chunks generados."""
    if not chunks:
        logger.warning("No se generaron chunks")
        return

    lengths = [len(c.page_content) for c in chunks]
    avg_len = sum(lengths) / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)

    # Contar documentos fuente únicos
    sources = {c.metadata.get("source_file", "?") for c in chunks}

    logger.info(
        f"Chunks generados: {len(chunks)} "
        f"(de {len(sources)} documentos)"
    )
    logger.info(
        f"Tamaño (chars): avg={avg_len:.0f}, min={min_len}, "
        f"max={max_len}, target={target_size}"
    )

    # Advertir si hay chunks muy grandes (puede indicar un problema)
    oversized = sum(1 for l in lengths if l > target_size * 1.5)
    if oversized > 0:
        logger.warning(
            f"{oversized} chunks exceden 1.5x el tamaño objetivo. "
            "Considerar ajustar separadores o chunk_size."
        )
