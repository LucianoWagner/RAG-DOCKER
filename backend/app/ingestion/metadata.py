"""
Metadata — Enriquecimiento de metadata para chunks del corpus Docker.

Responsabilidad:
- Asignar categoría temática (installation, troubleshooting, etc.)
- Detectar plataforma (windows, mac, linux, general)
- Extraer título del documento y encabezado de sección
- Clasificar tipo de documento (guide, reference, troubleshooting)

La metadata enriquecida permite:
1. Filtrado en la búsqueda (ej: solo chunks de troubleshooting)
2. Trazabilidad en las citas (indicar sección exacta)
3. Análisis de cobertura del corpus
"""

import re

from langchain_core.documents import Document
from loguru import logger

from app.models import DocCategory


# =============================================================================
# Reglas de clasificación
# =============================================================================

# Mapeo de keywords (en el path del archivo) a categorías.
# Se busca en orden; el primer match gana.
CATEGORY_RULES: list[tuple[str, DocCategory]] = [
    ("install", DocCategory.INSTALLATION),
    ("setup", DocCategory.INSTALLATION),
    ("get-started", DocCategory.GETTING_STARTED),
    ("getting-started", DocCategory.GETTING_STARTED),
    ("get-docker", DocCategory.GETTING_STARTED),
    ("introduction", DocCategory.GETTING_STARTED),
    ("concepts", DocCategory.GETTING_STARTED),
    ("troubleshoot", DocCategory.TROUBLESHOOTING),
    ("known-issues", DocCategory.TROUBLESHOOTING),
    ("workaround", DocCategory.TROUBLESHOOTING),
    ("reference", DocCategory.CLI_REFERENCE),
    ("cli", DocCategory.CLI_REFERENCE),
    ("commandline", DocCategory.CLI_REFERENCE),
    ("config", DocCategory.CONFIGURATION),
    ("daemon", DocCategory.CONFIGURATION),
    ("settings", DocCategory.CONFIGURATION),
]

# Mapeo de keywords a plataformas.
PLATFORM_RULES: list[tuple[str, str]] = [
    ("windows", "windows"),
    ("win", "windows"),
    ("mac", "mac"),
    ("macos", "mac"),
    ("apple", "mac"),
    ("linux", "linux"),
    ("ubuntu", "linux"),
    ("debian", "linux"),
    ("fedora", "linux"),
    ("centos", "linux"),
    ("rhel", "linux"),
]

# Mapeo de categoría a tipo de documento
_CATEGORY_TO_DOC_TYPE: dict[DocCategory, str] = {
    DocCategory.INSTALLATION: "guide",
    DocCategory.GETTING_STARTED: "guide",
    DocCategory.CLI_REFERENCE: "reference",
    DocCategory.TROUBLESHOOTING: "troubleshooting",
    DocCategory.CONFIGURATION: "guide",
}

# Regex para encontrar encabezados Markdown
_RE_HEADING = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


def enrich_metadata(chunks: list[Document]) -> list[Document]:
    """
    Enriquece la metadata de cada chunk con información estructurada.

    Cada chunk recibe:
    - category: Categoría temática del contenido
    - platform: Plataforma del SO (o "general")
    - doc_title: Título del documento (del frontmatter o path)
    - section_header: Encabezado más cercano dentro del chunk
    - doc_type: Tipo de documento (guide, reference, troubleshooting)

    Args:
        chunks: Lista de chunks con metadata básica.

    Returns:
        Los mismos chunks con metadata enriquecida.
    """
    logger.info(f"Enriqueciendo metadata de {len(chunks)} chunks")

    for chunk in chunks:
        source = chunk.metadata.get("source_file", "")
        content = chunk.page_content

        # Categoría temática
        chunk.metadata["category"] = detect_category(source, content).value

        # Plataforma
        chunk.metadata["platform"] = detect_platform(source, content)

        # Título del documento (prioridad: frontmatter > nombre archivo)
        if "doc_title" not in chunk.metadata or not chunk.metadata["doc_title"]:
            chunk.metadata["doc_title"] = _title_from_path(source)

        # Encabezado de sección más cercano dentro del chunk
        chunk.metadata["section_header"] = _extract_section_header(content)

        # Tipo de documento
        category = DocCategory(chunk.metadata["category"])
        chunk.metadata["doc_type"] = _CATEGORY_TO_DOC_TYPE.get(category, "guide")

    # Estadísticas de distribución
    _log_distribution(chunks)

    return chunks


def detect_category(source_file: str, content: str) -> DocCategory:
    """
    Detecta la categoría temática de un chunk.

    Estrategia: busca keywords primero en el path del archivo
    (más confiable), y si no encuentra, busca en el contenido.

    Args:
        source_file: Ruta relativa del archivo fuente.
        content: Contenido del chunk.

    Returns:
        DocCategory detectada. Default: GETTING_STARTED.
    """
    source_lower = source_file.lower()

    # Primero buscar en el path (más confiable)
    for keyword, category in CATEGORY_RULES:
        if keyword in source_lower:
            return category

    # Si no hay match en el path, buscar en contenido
    content_lower = content.lower()
    for keyword, category in CATEGORY_RULES:
        if keyword in content_lower:
            return category

    return DocCategory.GETTING_STARTED


def detect_platform(source_file: str, content: str) -> str:
    """
    Detecta la plataforma del chunk.

    Prioriza la detección por path del archivo. Si el path no indica
    plataforma, verifica si el contenido menciona una plataforma
    dominante (>= 3 menciones).

    Args:
        source_file: Ruta relativa del archivo fuente.
        content: Contenido del chunk.

    Returns:
        Plataforma detectada: "windows", "mac", "linux", o "general".
    """
    source_lower = source_file.lower()

    # Buscar en el path
    for keyword, platform in PLATFORM_RULES:
        if keyword in source_lower:
            return platform

    # Buscar en el contenido por frecuencia
    content_lower = content.lower()
    platform_counts: dict[str, int] = {}

    for keyword, platform in PLATFORM_RULES:
        count = content_lower.count(keyword)
        if count > 0:
            platform_counts[platform] = platform_counts.get(platform, 0) + count

    # Solo asignar si una plataforma aparece >= 3 veces (evitar falsos positivos)
    if platform_counts:
        best = max(platform_counts, key=platform_counts.get)  # type: ignore[arg-type]
        if platform_counts[best] >= 3:
            return best

    return "general"


def _title_from_path(source_file: str) -> str:
    """
    Genera un título legible a partir del path del archivo.

    Ejemplo: "desktop/install/windows-install.md" → "Windows Install"
    """
    if not source_file:
        return ""

    # Tomar el nombre del archivo sin extensión
    name = source_file.rsplit("/", 1)[-1]   # "windows-install.md"
    name = name.rsplit(".", 1)[0]           # "windows-install"
    name = name.replace("_index", "index")  # Hugo index files

    # Reemplazar guiones y underscores por espacios, capitalizar
    name = name.replace("-", " ").replace("_", " ")
    return name.title()


def _extract_section_header(content: str) -> str:
    """
    Extrae el primer encabezado Markdown del chunk.

    Busca el encabezado de mayor jerarquía (## > ### > ####).
    Si no hay encabezados, retorna string vacío.

    Args:
        content: Contenido del chunk.

    Returns:
        Texto del primer encabezado encontrado, sin el prefijo #.
    """
    matches = _RE_HEADING.findall(content)

    if not matches:
        return ""

    # Ordenar por jerarquía (menos # = mayor jerarquía)
    matches.sort(key=lambda m: len(m[0]))

    # Retornar el texto del encabezado de mayor jerarquía
    return matches[0][1].strip()


def _log_distribution(chunks: list[Document]) -> None:
    """Loggea la distribución de chunks por categoría y plataforma."""
    cat_counts: dict[str, int] = {}
    plat_counts: dict[str, int] = {}

    for chunk in chunks:
        cat = chunk.metadata.get("category", "unknown")
        plat = chunk.metadata.get("platform", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        plat_counts[plat] = plat_counts.get(plat, 0) + 1

    logger.info(f"Distribución por categoría: {dict(sorted(cat_counts.items()))}")
    logger.info(f"Distribución por plataforma: {dict(sorted(plat_counts.items()))}")
