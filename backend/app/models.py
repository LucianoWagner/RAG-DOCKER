"""
Modelos Pydantic del proyecto RAG.

Define las estructuras de datos que fluyen por todo el pipeline:
- Documentos procesados con metadata
- Respuestas con citas
- Resultados de evidencia
"""

from enum import Enum

from pydantic import BaseModel, Field


# =============================================================================
# Metadata de documentos
# =============================================================================

class DocCategory(str, Enum):
    """Categorías temáticas del corpus Docker."""
    INSTALLATION = "installation"
    GETTING_STARTED = "getting_started"
    CLI_REFERENCE = "cli_reference"
    TROUBLESHOOTING = "troubleshooting"
    CONFIGURATION = "configuration"


class ChunkMetadata(BaseModel):
    """Metadata enriquecida de cada chunk de documentación."""
    source_file: str = Field(description="Ruta del archivo fuente")
    doc_title: str = Field(default="", description="Título del documento")
    section_header: str = Field(default="", description="Encabezado de sección")
    category: DocCategory = Field(default=DocCategory.GETTING_STARTED)
    platform: str = Field(default="general", description="Plataforma: windows, mac, linux, general")
    doc_type: str = Field(default="guide", description="Tipo: guide, reference, troubleshooting")
    chunk_index: int = Field(default=0, description="Índice del chunk en el documento")
    total_chunks: int = Field(default=0, description="Total de chunks del documento")


# =============================================================================
# Citas y fuentes
# =============================================================================

class SourceCitation(BaseModel):
    """Una cita individual a un fragmento del corpus."""
    citation_id: int = Field(description="Número de referencia [Fuente N]")
    source_file: str = Field(description="Archivo fuente")
    doc_title: str = Field(default="", description="Título del documento")
    section_header: str = Field(default="", description="Sección")
    relevant_fragment: str = Field(description="Fragmento textual utilizado")
    relevance_score: float = Field(default=0.0, description="Score de relevancia")


# =============================================================================
# Verificación de evidencia
# =============================================================================

class EvidenceVerdict(str, Enum):
    """Veredicto sobre la suficiencia de la evidencia recuperada."""
    SUFFICIENT = "sufficient"
    LOW_CONFIDENCE = "low_confidence"
    INSUFFICIENT = "insufficient"


class EvidenceResult(BaseModel):
    """Resultado de la evaluación de evidencia."""
    verdict: EvidenceVerdict
    top_score: float = Field(description="Score del chunk más relevante")
    relevant_count: int = Field(description="Cantidad de chunks con score > umbral")
    details: str = Field(default="", description="Explicación del veredicto")


# =============================================================================
# Respuesta del pipeline
# =============================================================================

class QueryRequest(BaseModel):
    """Solicitud de consulta al pipeline RAG."""
    question: str = Field(description="Pregunta del usuario")


class RAGResponse(BaseModel):
    """Respuesta completa del pipeline RAG con citas y trazabilidad."""
    answer: str = Field(description="Texto de la respuesta con citas inline")
    sources: list[SourceCitation] = Field(default_factory=list, description="Fuentes citadas")
    evidence: EvidenceResult = Field(description="Resultado de verificación de evidencia")
    retrieval_metadata: dict = Field(default_factory=dict, description="Metadata del proceso")
