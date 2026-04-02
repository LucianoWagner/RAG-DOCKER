# Arquitectura del Sistema RAG

## Diagrama General

```
┌─────────────────────────────────────────────────────────────────┐
│                    Capa de Presentación                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              OpenWebUI (puerto 3000)                     │  │
│  │         Frontend conversacional tipo ChatGPT             │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │ HTTP / OpenAI-compatible API         │
├─────────────────────────┼───────────────────────────────────────┤
│                    Capa de Orquestación                        │
│  ┌──────────────────────▼───────────────────────────────────┐  │
│  │         Backend RAG (FastAPI + LangChain)                │  │
│  │                   (puerto 8080)                          │  │
│  │                                                          │  │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌───────────┐ │  │
│  │  │ Ingesta │→ │ Retrieval│→ │Evidence │→ │Generación │ │  │
│  │  │         │  │ Híbrido  │  │  Check  │  │ + Citas   │ │  │
│  │  └─────────┘  └──────────┘  └─────────┘  └───────────┘ │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                      │
├─────────────────────────┼───────────────────────────────────────┤
│                    Capa de Modelos                             │
│  ┌──────────────────────▼───────────────────────────────────┐  │
│  │              Ollama (puerto 11434)                       │  │
│  │  ┌────────────────┐    ┌───────────────────┐            │  │
│  │  │ LLM            │    │ Embeddings         │            │  │
│  │  │ llama3.1:8b    │    │ nomic-embed-text   │            │  │
│  │  └────────────────┘    └───────────────────┘            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                    Capa de Almacenamiento                      │
│  ┌────────────────────────┐   ┌────────────────────────┐      │
│  │ ChromaDB (puerto 8000) │   │ Corpus Markdown        │      │
│  │ Base vectorial         │   │ (volumen Docker)       │      │
│  │ (volumen persistente)  │   │                        │      │
│  └────────────────────────┘   └────────────────────────┘      │
└────────────────────────────────────────────────────────────────┘
```

## Módulos del Backend

| Módulo | Archivos | Responsabilidad |
|:---|:---|:---|
| **Ingesta** | `ingestion/` | Carga, limpieza, chunking, metadata, indexación |
| **Retrieval** | `retrieval/` | Búsqueda semántica, BM25, híbrida, reranking |
| **Generación** | `generation/` | Evidence check, prompts, LLM, citas |
| **Pipeline** | `pipeline.py` | Orquesta todo el flujo RAG |

## Flujo de Datos

1. **Ingesta** (una sola vez o al actualizar corpus)
   - Markdown → Preprocesamiento → Chunking → Metadata → Embeddings → ChromaDB

2. **Consulta** (cada pregunta del usuario)
   - Pregunta → Embedding → Búsqueda Híbrida → Reranking → Evidence Check → LLM → Respuesta con citas
