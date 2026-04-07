# Implementación de Traducción Automática en el RAG

A continuación se detallan los cambios estructurales que se aplicaron en el backend para lograr que el sistema detecte si una pregunta está en español, la traduzca al inglés para su procesamiento, y luego vuelva a traducir la respuesta técnica generada al español.

## 1. Creación del módulo `translator.py`
Se creó un nuevo archivo para aislar la lógica de traducción y detección de idioma.

**Archivo:** `backend/app/generation/translator.py`
**Funciones añadidas:**
- `detect_spanish(llm, text)`: Usa un `SystemMessage` con el LLM para responder con un simple "YES" o "NO" dependiendo de si detecta que la entrada del usuario está en español.
- `translate_to_english(llm, text)`: Toma la pregunta detectada en español y la traduce al inglés usando el LLM (esencial ya que los chunks vectorizados y el `BM25` funcionan mejor con los términos en inglés nativos de la documentación técnica).
- `translate_to_spanish(llm, text)`: Traduce la respuesta generada desde el inglés al español técnico, asegurándose de no alterar comandos vitales de docker (e.g. *containers*, *volumes*, *docker compose*).

## 2. Soporte en Modelos de Metadatos
Para evitar conflictos de Tipado (Typos) y de variables no soportadas debido al cambio a objetos Pydantic, se añadieron los campos necesarios al metadato del pipeline.

**Archivo:** `backend/app/models.py`
**Cambios:**
Se añadieron los siguientes atributos nuevos a la clase `RetrievalMetadata`:
```python
class RetrievalMetadata(BaseModel):
    # ...
    original_question: str | None = Field(default=None, description="Pregunta antes de traducir (si aplica)")
    translated_question: str | None = Field(default=None, description="Pregunta traducida utilizada (si aplica)")
    status: str | None = Field(default=None, description="Estado del retrieval (e.g., abstained)")
    # ...
```

## 3. Integración en el Pipeline Principal
Se incrustó completamente la capacidad de traducción dentro del orquestador.

**Archivo:** `backend/app/pipeline.py`
**Cambios en el método `RAGPipeline.run()`:**

1. **Detección Previa:** Al inicio del flujo, se detecta el lenguaje usando `detect_spanish()`.
2. **Traducción Inicial (Forward):** Si resulta verdadero (el usuario preguntó en español), la pregunta se reasigna a su versión en inglés pasando por `translate_to_english()`. Esta versión en inglés es la que transcurre a lo largo del Retrieval Híbrido y el Reranking.
3. **Manejo de Casos en Blanco (Abstenciones):** Si la evidencia es insuficiente y se utiliza una abstención pre-programada, esta abstención también es pasada por el traductor al español antes de ser devuelta.
4. **Traducción Final (Backward):** Justo antes del `return response`, se re-evalúa si la pregunta original era en español. Si es afirmativo, se toma la `response.answer` generada y se le aplica `translate_to_spanish()` para devolverle al usuario un resultado en el mismo lenguaje del input.
5. **Registro de Trazabilidad:** Se actualizan formalmente los valores `response.retrieval_metadata.original_question` y `response.retrieval_metadata.translated_question` usando notación de puntos (Object Attributes) para el registro de auditoría.
