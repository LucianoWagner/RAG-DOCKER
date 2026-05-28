# Flujo completo de `backend/app/pipeline.py` ante una query de usuario

Este documento explica, de punta a punta, que pasa en el proyecto cuando un usuario envia una pregunta al asistente RAG de soporte Docker. Esta basado en `AGENTS.md` y en el codigo actual de `backend/app/pipeline.py` y sus modulos relacionados.

El objetivo es que puedas repasar el flujo completo sin tener que saltar archivo por archivo.

## 1. Idea general del sistema

El proyecto es un asistente RAG especializado en documentacion oficial de Docker.

RAG significa `Retrieval-Augmented Generation`: antes de pedirle al LLM que responda, el sistema busca fragmentos relevantes en una base documental, los mete como contexto en el prompt, y obliga al modelo a responder usando solo esos fragmentos.

En este proyecto el flujo principal es:

```text
Usuario
  -> FastAPI
  -> RAGPipeline.run(question)
  -> deteccion/traduccion de idioma
  -> router de intencion
  -> retrieval hibrido: Chroma semantico + BM25 lexico
  -> reranking con FlashRank
  -> evidence checker
  -> prompt con contexto
  -> Groq LLM
  -> parseo de citas
  -> traduccion final si la pregunta era en espanol
  -> RAGResponse
```

Los componentes principales son:

| Componente | Archivo | Rol |
|---|---|---|
| API HTTP | `backend/app/main.py` | Recibe requests y llama al pipeline |
| Orquestador | `backend/app/pipeline.py` | Coordina todo el flujo RAG |
| Vector store | `backend/app/retrieval/vector_store.py` | ChromaDB + embeddings de Ollama |
| BM25 | `backend/app/retrieval/bm25_retriever.py` | Busqueda lexica por terminos exactos |
| Fusion hibrida | `backend/app/retrieval/hybrid.py` | Une semantico y BM25 con RRF |
| Reranker | `backend/app/retrieval/reranker.py` | Reordena chunks con FlashRank |
| Evidencia | `backend/app/generation/evidence_checker.py` | Decide si hay suficiente evidencia |
| Prompt | `backend/app/generation/prompt_templates.py` | Arma contexto y mensajes del LLM |
| Generacion | `backend/app/generation/generator.py` | Invoca Groq y arma `RAGResponse` |
| Traduccion | `backend/app/generation/translator.py` | ES -> EN y EN -> ES |
| Router | `backend/app/generation/router.py` | Decide `RAG` vs `CHITCHAT` |

## 2. Como entra la query al backend

Hay dos formas principales:

1. `POST /query`
2. `POST /v1/chat/completions`, compatible con OpenAI, usado por OpenWebUI

El caso mas importante para OpenWebUI esta en `backend/app/main.py`:

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    question = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            question = msg.content
            break

    if not question:
        question = "Hola"

    pipeline = get_pipeline()
    rag_response = pipeline.run(question)
```

Puntos clave:

- OpenWebUI manda una lista de mensajes.
- El backend toma el ultimo mensaje con `role == "user"`.
- Si no encuentra ninguno, usa `"Hola"`.
- Llama a `get_pipeline()`.
- `get_pipeline()` esta cacheado con `@lru_cache`, por eso el pipeline se inicializa una sola vez por proceso.

```python
@lru_cache()
def get_pipeline():
    from app.pipeline import RAGPipeline
    return RAGPipeline()
```

Esto es importante: la primera query suele tardar mas porque inicializa Chroma, BM25, FlashRank y el LLM. Las siguientes queries reutilizan la misma instancia.

## 3. Inicializacion de `RAGPipeline`

Cuando se crea `RAGPipeline()`, se ejecuta `__init__()`:

```python
class RAGPipeline:
    def __init__(self):
        self.settings = get_settings()
        logger.info("Inicializando RAG Pipeline...")

        from app.retrieval.vector_store import get_vector_store, get_semantic_retriever
        from app.retrieval.bm25_retriever import create_bm25_retriever
        from app.retrieval.hybrid import create_hybrid_retriever
        from app.generation.generator import get_llm

        self.vector_store = get_vector_store()
        self.semantic_retriever = get_semantic_retriever()

        res = self.vector_store._collection.get(include=["documents", "metadatas"])
        chunks = []
        if res and res["documents"]:
            for text, meta in zip(res["documents"], res["metadatas"]):
                chunks.append(Document(page_content=text, metadata=meta))

        if chunks:
            self.bm25_retriever = create_bm25_retriever(chunks)
            self.hybrid_retriever = create_hybrid_retriever(
                self.semantic_retriever,
                self.bm25_retriever,
            )
        else:
            self.bm25_retriever = None
            self.hybrid_retriever = self.semantic_retriever

        self.llm = get_llm()
```

La inicializacion hace cuatro cosas:

1. Crea el cliente de ChromaDB.
2. Crea el retriever semantico.
3. Reconstruye BM25 en memoria leyendo todos los chunks ya indexados en Chroma.
4. Crea el LLM de Groq.

### 3.1. ChromaDB y embeddings

El vector store se crea en `backend/app/retrieval/vector_store.py`:

```python
def get_embedding_function() -> OllamaEmbeddings:
    settings = get_settings()
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )
```

```python
def get_vector_store() -> Chroma:
    settings = get_settings()
    return Chroma(
        client=get_chroma_client(),
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embedding_function(),
    )
```

Configuracion relevante:

```python
embedding_model: str = "nomic-embed-text"
ollama_base_url: str = "http://127.0.0.1:11434"
chroma_host: str = "localhost"
chroma_port: int = 8000
chroma_collection_name: str = "docker_docs"
```

Esto significa:

- Los embeddings los genera Ollama con `nomic-embed-text`.
- Los vectores se guardan en ChromaDB.
- La coleccion se llama `docker_docs`.

### 3.2. Retriever semantico

El retriever semantico se crea asi:

```python
def get_semantic_retriever() -> VectorStoreRetriever:
    settings = get_settings()
    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_kwargs={"k": settings.top_k}
    )
```

Con la configuracion actual:

```python
top_k: int = 30
```

Entonces, cuando se invoca el retriever semantico:

```python
self.semantic_retriever.invoke(question)
```

LangChain hace, conceptualmente:

1. Convierte la pregunta en un vector con `nomic-embed-text`.
2. Busca en Chroma los chunks con embeddings mas parecidos.
3. Devuelve los `k = 30` documentos mas cercanos.

Detalle importante: el codigo no sobreescribe explicitamente la metrica de distancia/similitud de Chroma. Usa el comportamiento default de `langchain_chroma.Chroma` y ChromaDB.

## 4. Como se preparan los chunks antes de existir en Chroma

La busqueda semantica depende muchisimo de como fueron creados los chunks. Eso ocurre en el pipeline de ingesta, no en `pipeline.py`.

Archivo: `backend/app/ingestion/run.py`

```python
documents = load_markdown_files(CORPUS_DIR)
documents = preprocess_documents(documents)
chunks = chunk_documents(documents)
chunks = enrich_metadata(chunks)

from app.retrieval.vector_store import index_documents
indexed_count = index_documents(chunks, force=force)
```

### 4.1. Limpieza previa

`preprocessor.py` limpia la documentacion Markdown de Docker:

- Extrae frontmatter YAML.
- Elimina shortcodes Hugo.
- Elimina HTML residual.
- Normaliza espacios.
- Preserva bloques de codigo.
- Filtra documentos demasiado cortos.

Esto importa porque los embeddings serian peores si indexaran artefactos de Hugo, HTML o basura no semantica.

### 4.2. Chunking Markdown-aware

`chunker.py` usa `RecursiveCharacterTextSplitter`:

```python
MARKDOWN_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n```",
    "\n\n",
    "\n",
    ". ",
    " ",
]
```

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=size,
    chunk_overlap=overlap,
    separators=MARKDOWN_SEPARATORS,
    length_function=len,
    is_separator_regex=False,
    keep_separator=True,
)
```

Configuracion:

```python
chunk_size: int = 800
chunk_overlap: int = 200
```

Tecnica aplicada:

- Divide primero por secciones Markdown (`##`, `###`, `####`).
- Si una seccion es muy larga, baja a bloques de codigo, parrafos, lineas, oraciones y palabras.
- Usa overlap de 200 caracteres para que un chunk no pierda contexto por estar justo en el borde.
- Mantiene separadores para preservar encabezados dentro del chunk.

Esto es clave para RAG: un chunk demasiado grande mete ruido; uno demasiado chico pierde contexto. Este proyecto apunta a chunks medianos, con contexto local.

### 4.3. Metadata enriquecida

`metadata.py` agrega campos utiles:

```python
chunk.metadata["category"] = detect_category(source, content).value
chunk.metadata["platform"] = detect_platform(source, content)
chunk.metadata["doc_title"] = _title_from_path(source)
chunk.metadata["section_header"] = _extract_section_header(content)
chunk.metadata["doc_type"] = ...
```

Ejemplos:

| Campo | Para que sirve |
|---|---|
| `source_file` | Saber de que archivo salio el chunk |
| `doc_title` | Mostrar titulo humano |
| `section_header` | Mostrar seccion especifica |
| `category` | Instalacion, troubleshooting, CLI, etc. |
| `platform` | Windows, Linux, Mac o general |
| `chunk_index` | Posicion dentro del documento original |
| `total_chunks` | Cantidad total de chunks de ese documento |

La metadata no se usa hoy para filtrar en la busqueda principal, pero si se usa para trazabilidad, citas y auditoria.

### 4.4. Indexacion con embeddings

`vector_store.index_documents()` llama:

```python
vector_store.add_documents(chunks)
```

Ese metodo:

1. Toma cada `Document`.
2. Genera su embedding con `OllamaEmbeddings`.
3. Guarda texto, metadata y vector en ChromaDB.

La indexacion no ocurre en cada pregunta. Ocurre cuando se corre la ingesta.

## 5. Ejecucion de `RAGPipeline.run(question)`

El metodo central es:

```python
def run(self, question: str) -> RAGResponse:
    from app.retrieval.hybrid import retrieve
    from app.retrieval.reranker import rerank_documents
    from app.generation.evidence_checker import check_evidence, get_abstention_response
    from app.generation.prompt_templates import format_context, build_messages
    from app.generation.generator import generate_response
    from app.generation.translator import detect_spanish, translate_to_english, translate_to_spanish
    from app.generation.router import route_query, get_chitchat_response
```

`pipeline.py` no implementa toda la logica internamente. Orquesta modulos especializados.

## 6. Paso 0: deteccion de idioma y traduccion

Primero detecta si la pregunta esta en espanol:

```python
is_spanish = detect_spanish(self.llm, question)
original_question = question

if is_spanish:
    question = translate_to_english(self.llm, question)
    logger.info(f"Pregunta traducida al ingles para procesamiento: {question}")
```

`translator.py` usa `langdetect`:

```python
def detect_spanish(llm, text: str) -> bool:
    lang = detect(text)
    is_spanish = lang == "es"
    return is_spanish
```

Si detecta espanol, traduce la query a ingles con Google Translate:

```python
translation = GoogleTranslator(source="es", target="en").translate(text)
```

Por que se traduce a ingles:

- El corpus de Docker esta principalmente en ingles.
- Los embeddings y el retrieval funcionan mejor si query y documentos estan en el mismo idioma.
- La generacion tambien se fuerza a ingles y luego se traduce de vuelta si hace falta.

Detalle fino:

- `original_question` conserva la pregunta original del usuario.
- `question` pasa a ser la version de trabajo, usualmente en ingles.
- El router y el retrieval trabajan sobre `question`, no sobre `original_question`.

Ejemplo:

```text
Original: "Como instalo Docker en Windows?"
Trabajo:  "How do I install Docker on Windows?"
```

## 7. Paso 1: semantic router, `RAG` o `CHITCHAT`

Despues de la traduccion, el pipeline clasifica la intencion:

```python
intent = route_query(self.llm, question)

if intent == "CHITCHAT":
    chitchat_answer = get_chitchat_response(self.llm, original_question)
    return RAGResponse(...)
```

El router esta en `backend/app/generation/router.py`.

### 7.1. Heuristicas locales

Primero revisa keywords tecnicas:

```python
_RAG_KEYWORDS = [
    "docker", "container", "compose", "dockerfile", "image", "volume",
    "network", "build", "run", "pull", "push", "swarm", "daemon",
    "install", "instalar", "configurar", "configure", "error", "log",
    "port", "puerto", "deploy", "kubernetes", "k8s", "registry",
    "command", "comando", "cli", "terminal", "sudo", "wsl",
]
```

Si alguna aparece:

```python
if any(kw in text for kw in _RAG_KEYWORDS):
    return "RAG"
```

Luego revisa patrones de conversacion trivial:

```python
_CHITCHAT_PATTERNS = [
    re.compile(r"^\s*(hola|hello|hi|hey|buenas|...)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(gracias|thanks|thank\s*you|...)\s*[!.?]*\s*$", re.IGNORECASE),
    ...
]
```

Si matchea, devuelve:

```python
return "CHITCHAT"
```

### 7.2. Fallback con LLM

Si las reglas no alcanzan, llama al LLM con un prompt few-shot:

```python
messages = [
    SystemMessage(content=f"{_ROUTER_SYSTEM_PROMPT}\n\nExamples:\n{examples_text}"),
    HumanMessage(content=question),
]

response = llm.invoke(messages)
result = response.content.strip().upper()
```

El LLM debe responder solo:

```text
RAG
```

o:

```text
CHITCHAT
```

Si hay error o respuesta rara, el router default es `RAG`. Esa decision es conservadora: ante la duda, busca en la documentacion.

### 7.3. Si es CHITCHAT

El pipeline no busca en Chroma ni usa BM25:

```python
return RAGResponse(
    answer=chitchat_answer,
    sources=[],
    evidence=EvidenceResult(
        verdict=EvidenceVerdict.SUFFICIENT,
        top_score=1.0,
        relevant_count=0,
        details="Chit-chat: no se requiere busqueda en documentacion.",
    ),
    retrieval_metadata=RetrievalMetadata(
        question=original_question,
        original_question=original_question,
        translated_question=question if is_spanish else None,
        chunks_used=0,
        status="chit-chat",
    ),
)
```

Resultado:

- `sources = []`
- `chunks_used = 0`
- `status = "chit-chat"`
- No hay retrieval.

## 8. Paso 2: retrieval hibrido

Si la intencion es `RAG`, se ejecuta:

```python
retrieved_chunks = (
    retrieve(self.hybrid_retriever, question)
    if self.bm25_retriever
    else self.hybrid_retriever.invoke(question)
)
```

Hay dos casos:

1. Si BM25 se pudo construir, usa retrieval hibrido.
2. Si no habia chunks para BM25, usa solo el retriever semantico.

El caso normal es el hibrido.

## 9. Busqueda semantica: que hace exactamente

La busqueda semantica es la parte basada en embeddings.

Codigo:

```python
self.semantic_retriever = get_semantic_retriever()
```

```python
return vector_store.as_retriever(
    search_kwargs={"k": settings.top_k}
)
```

Cuando se invoca con una pregunta:

```python
semantic_docs = semantic_retriever.invoke(query)
```

ocurre este proceso conceptual:

```text
query en texto
  -> OllamaEmbeddings("nomic-embed-text")
  -> vector numerico de la query
  -> ChromaDB compara ese vector contra los vectores de chunks
  -> devuelve los 30 chunks mas cercanos
```

Ejemplo conceptual:

```text
"How do I persist data in Docker?"
  -> [0.021, -0.391, 0.114, ...]
  -> Chroma busca chunks cercanos
  -> devuelve chunks sobre volumes, bind mounts, persistencia, etc.
```

Tecnica aplicada:

- Bi-encoder retrieval.
- Query y documentos se vectorizan por separado.
- La similitud se calcula en el espacio vectorial.
- Es buena para sinonimos y significado general.

Ejemplo de fortaleza:

```text
Query: "How can I keep container data after restart?"
```

Aunque el usuario no diga literalmente `volume`, el embedding puede acercarlo a chunks sobre `volumes` o `persisting container data`.

Limite:

- Puede fallar con strings exactos: flags, comandos, errores literales.
- Por eso se combina con BM25.

## 10. Busqueda lexica BM25

BM25 esta en `backend/app/retrieval/bm25_retriever.py`:

```python
def create_bm25_retriever(chunks: list[Document]) -> BM25Retriever:
    settings = get_settings()
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = settings.top_k
    return bm25_retriever
```

BM25 no usa embeddings. Usa terminos del texto.

Tecnica aplicada:

- Ranking lexico por frecuencia de terminos.
- Premia documentos que contienen palabras importantes de la query.
- Penaliza palabras demasiado comunes.

Ejemplo donde BM25 ayuda:

```text
"Cannot connect to the Docker daemon"
"docker compose up"
"--mount"
"ENTRYPOINT"
"exit code 137"
```

Estos strings exactos suelen ser mejor capturados por BM25 que por embeddings.

Detalle importante:

- BM25 se reconstruye en memoria al iniciar `RAGPipeline`.
- Lee documentos desde Chroma:

```python
res = self.vector_store._collection.get(include=["documents", "metadatas"])
```

- No se persiste como indice aparte.

## 11. Fusion hibrida con RRF

El archivo `backend/app/retrieval/hybrid.py` combina resultados semanticos y lexicos.

```python
class CustomHybridRetriever:
    def invoke(self, query: str) -> list[Document]:
        all_results = []
        for r in self.retrievers:
            all_results.append(r.invoke(query))
```

Tiene dos retrievers:

```python
retrievers=[semantic_retriever, bm25_retriever]
weights=[settings.semantic_weight, settings.bm25_weight]
```

Configuracion:

```python
semantic_weight: float = 0.4
bm25_weight: float = 0.6
```

Es decir:

- Semantico: 40%
- BM25: 60%

### 11.1. Formula RRF

RRF significa `Reciprocal Rank Fusion`.

Codigo:

```python
for weight, docs in zip(self.weights, all_results):
    for rank, doc in enumerate(docs):
        key = doc.page_content
        if key not in doc_map:
            doc_map[key] = doc
            rrf_score[key] = 0.0

        rrf_score[key] += weight * (1.0 / (60 + rank))
```

La formula usada es:

```text
score(doc) += weight * 1 / (60 + rank)
```

Como `rank` arranca en 0:

```text
primer resultado: weight * 1/60
segundo resultado: weight * 1/61
tercer resultado: weight * 1/62
```

Despues ordena:

```python
sorted_keys = sorted(rrf_score.keys(), key=lambda x: rrf_score[x], reverse=True)
return [doc_map[k] for k in sorted_keys]
```

### 11.2. Por que RRF es util aca

Supongamos:

```text
Chunk A:
  ranking semantico: 1
  ranking BM25: 12

Chunk B:
  ranking semantico: 8
  ranking BM25: 1
```

RRF permite que ambos compitan aunque vengan de retrievers distintos.

Ventajas:

- No necesita que ambos retrievers usen scores comparables.
- Solo usa posiciones de ranking.
- Es estable y simple.
- Funciona bien para fusionar embeddings + keywords.

### 11.3. Deduplicacion

La key de deduplicacion es:

```python
key = doc.page_content
```

Si el mismo chunk aparece por Chroma y por BM25, se fusiona en una sola entrada.

Nota: si dos chunks distintos tuvieran exactamente el mismo texto, este metodo los trataria como duplicados.

## 12. Paso 3: reranking con FlashRank

Luego del retrieval hibrido:

```python
reranked_chunks = rerank_documents(question, retrieved_chunks)
```

Archivo: `backend/app/retrieval/reranker.py`

```python
passages = []
for i, doc in enumerate(documents):
    passages.append({
        "id": i,
        "text": doc.page_content,
        "meta": doc.metadata
    })

rerank_req = RerankRequest(query=query, passages=passages)
results = _ranker.rerank(rerank_req)
```

Luego toma solo `top_n`:

```python
top_n = settings.rerank_top_n

for rank, res in enumerate(results[:top_n]):
    doc = Document(page_content=res["text"], metadata=res.get("meta", {}))
    doc.metadata["rerank_score"] = res.get("score", 0.0)
    doc.metadata["rerank_position"] = rank + 1
    ranked_docs.append(doc)
```

Configuracion:

```python
rerank_top_n: int = 10
```

Tecnica aplicada:

- El retrieval inicial es bi-encoder: query y documentos se representan por separado.
- El reranker es cross-encoder: evalua query y documento juntos.
- Es mas preciso, pero mas costoso.
- Por eso se aplica solo sobre los documentos ya recuperados, no sobre todo el corpus.

Resultado:

- Se queda con los 10 mejores chunks.
- Cada chunk recibe:

```python
rerank_score
rerank_position
```

Importante: el evidence checker usa `rerank_score`, no el score original de Chroma ni el de BM25.

## 13. Paso 4: evidence checker

Despues del reranking:

```python
evidence = check_evidence(question, reranked_chunks)
```

El evidence checker decide si el sistema tiene suficiente evidencia para responder.

Archivo: `backend/app/generation/evidence_checker.py`

### 13.1. Si no hay chunks

```python
if not reranked_chunks:
    return EvidenceResult(
        verdict=EvidenceVerdict.INSUFFICIENT,
        top_score=0.0,
        relevant_count=0,
        details="No se encontraron documentos recuperados."
    )
```

### 13.2. Score del mejor chunk

```python
top_chunk = reranked_chunks[0]
top_score = top_chunk.metadata.get("rerank_score", 0.0)
```

Si el mejor chunk tiene score bajo:

```python
if top_score < settings.min_top_score:
    return EvidenceResult(
        verdict=EvidenceVerdict.INSUFFICIENT,
        top_score=top_score,
        relevant_count=relevant_count,
        details=f"Mejor match tiene score {top_score:.2f} < {settings.min_top_score}"
    )
```

Configuracion:

```python
min_top_score: float = 0.3
```

Si el mejor chunk no llega a 0.3, el pipeline no genera respuesta con el LLM.

### 13.3. Cantidad de chunks relevantes

```python
relevant_chunks = [
    c for c in reranked_chunks
    if c.metadata.get("rerank_score", 0.0) >= settings.relevance_threshold
]
relevant_count = len(relevant_chunks)
```

Configuracion:

```python
relevance_threshold: float = 0.25
min_relevant_chunks: int = 2
```

Si hay menos de 2 chunks relevantes, el verdict es `LOW_CONFIDENCE`.

### 13.4. Dispersion de scores

```python
last_score = reranked_chunks[-1].metadata.get("rerank_score", 0.0)
score_spread = top_score - last_score

if top_score < 0.85 and score_spread < 0.05:
    return EvidenceResult(
        verdict=EvidenceVerdict.LOW_CONFIDENCE,
        ...
    )
```

La idea:

- Si todos los chunks tienen scores parecidos y no muy altos, tal vez no hay un ganador real.
- Eso puede indicar ruido.

### 13.5. Veredictos posibles

| Verdict | Que significa | Que hace el pipeline |
|---|---|---|
| `SUFFICIENT` | Hay evidencia fuerte | Genera respuesta |
| `LOW_CONFIDENCE` | Hay evidencia, pero debil | Igual genera respuesta |
| `INSUFFICIENT` | No hay evidencia minima | Se abstiene |

Detalle importante:

```python
if evidence.verdict == EvidenceVerdict.INSUFFICIENT:
    response_text = get_abstention_response()
    ...
    return RAGResponse(...)
```

Solo `INSUFFICIENT` corta el flujo. `LOW_CONFIDENCE` continua hacia generacion.

## 14. Abstencion

Si la evidencia es insuficiente:

```python
response_text = get_abstention_response()

if is_spanish:
    response_text = translate_to_spanish(self.llm, response_text)

return RAGResponse(
    answer=response_text,
    sources=[],
    evidence=evidence,
    retrieval_metadata=RetrievalMetadata(
        question=original_question,
        original_question=original_question,
        translated_question=question if is_spanish else None,
        chunks_used=0,
        status="abstained",
    ),
)
```

Resultado:

- No se llama al generador.
- No se inventa respuesta.
- `sources = []`.
- `status = "abstained"`.

Esta es una proteccion anti-alucinacion.

## 15. Paso 5: armado del contexto y prompt

Si la evidencia no es `INSUFFICIENT`, el pipeline arma contexto:

```python
context = format_context(reranked_chunks)
messages = build_messages(question, context)
response = generate_response(question, reranked_chunks, evidence, messages, llm=self.llm)
```

### 15.1. Formato del contexto

`format_context()` convierte los chunks en bloques numerados:

```python
chunk_str = (
    f"[Source {n}]\n"
    f"Document: {title}\n"
    f"Section: {section}\n"
    f"---\n"
    f"{chunk.page_content}\n"
    f"---"
)
```

Ejemplo:

```text
[Source 1]
Document: Persisting container data
Section: Volumes
---
Texto del chunk...
---

[Source 2]
Document: Bind mounts
Section: Sharing files
---
Texto del chunk...
---
```

La numeracion importa porque el LLM debe citar con `[Source N]`.

### 15.2. System prompt anti-alucinacion

El prompt principal dice:

```text
You are a strictly grounded technical assistant specialized in Docker.
Your sole purpose is to answer questions using ONLY the provided documentation context.
```

Reglas relevantes:

```text
1. You are FORBIDDEN from using any external or pre-trained knowledge.
2. If a specific detail ... is not EXPLICITLY written ... DO NOT INVENT IT.
3. Every single assertion ... MUST be directly verifiable against the context snippets.
5. Cite your sources using the format [Source N] ...
6. YOUR FINAL RESPONSE MUST ALWAYS BE IN ENGLISH.
```

El user prompt:

```python
USER_PROMPT = """QUESTION: {question}

Answer based exclusively on the provided documentation fragments.
Include citations [Source N] for every key statement. All output must be in English."""
```

Esto hace que la generacion base sea siempre en ingles. Si la pregunta original era en espanol, la traduccion final ocurre despues.

## 16. Paso 6: generacion con Groq

El LLM se crea en `generator.py`:

```python
def get_llm() -> ChatGroq:
    settings = get_settings()
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY no encontrada. Agregala a tu archivo .env")

    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=settings.groq_api_key,
        temperature=0.0,
    )
```

Detalles:

- Usa Groq.
- Modelo: `llama-3.3-70b-versatile`.
- Temperatura `0.0` para minimizar creatividad.
- Requiere `GROQ_API_KEY`.

La generacion:

```python
response = llm.invoke(messages)
answer_text = response.content
```

Luego arma la respuesta:

```python
return RAGResponse(
    answer=answer_text,
    sources=citations,
    evidence=evidence,
    retrieval_metadata=RetrievalMetadata(
        question=question,
        chunks_used=len(context_chunks),
        chunks_metadata=[
            ChunkMetadata(**chunk.metadata, chunk_text=chunk.page_content)
            for chunk in context_chunks
        ]
    ),
)
```

## 17. Parseo de citas

Despues de generar texto, se extraen citas:

```python
citations = parse_citations(answer_text, context_chunks)
```

El parser busca:

```python
matches = re.findall(r"\[Source (\d+)\]", answer_text, re.IGNORECASE)
```

Si el LLM escribio:

```text
Docker volumes are used to persist data. [Source 2]
```

Entonces:

- Extrae `2`.
- Busca el chunk `context_chunks[1]`.
- Arma un `SourceCitation`.

```python
SourceCitation(
    citation_id=n,
    source_file=meta.get("source_file", "unknown"),
    doc_title=meta.get("doc_title", "Unspecified"),
    section_header=meta.get("section_header", "General"),
    relevant_fragment=chunk.page_content,
    relevance_score=meta.get("relevance_score", 0.0)
)
```

Detalle importante:

- El parser espera `[Source N]`, no `[Fuente N]`.
- `relevance_score` sale de `meta.get("relevance_score", 0.0)`.
- Hoy el reranker guarda `rerank_score`, no `relevance_score`, asi que ese campo puede quedar en `0.0` aunque el chunk haya sido relevante.

## 18. Paso 7: traduccion final al espanol

Si la pregunta original era en espanol:

```python
if is_spanish:
    response.answer = translate_to_spanish(self.llm, response.answer)
    response.retrieval_metadata.translated_question = question
```

La traduccion final usa el LLM, no Google Translate:

```python
messages = [
    SystemMessage(content=(
        "You are a professional technical translator. Translate the following text "
        "from English to Spanish. CRITICAL RULES:\n"
        "1. NEVER translate code blocks, commands, file paths, or environment variables.\n"
        "2. Keep Docker terminology in English: containers, volumes, images, compose, "
        ...
    )),
    HumanMessage(content=text)
]

response = llm.invoke(messages)
```

La razon:

- La respuesta puede contener comandos.
- Puede contener paths.
- Puede contener variables de entorno.
- Puede contener terminos Docker que conviene preservar.

Ejemplo de algo que no se debe romper:

```bash
docker run -d -p 80:80 nginx
```

Por eso la query ES -> EN usa Google Translate, pero la respuesta EN -> ES usa LLM.

## 19. Metadata final

Antes de devolver:

```python
response.retrieval_metadata.original_question = original_question
response.retrieval_metadata.question = original_question
return response
```

Esto hace que hacia afuera se vea la pregunta original del usuario, no la traduccion interna.

La respuesta final tiene esta estructura:

```python
class RAGResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    evidence: EvidenceResult
    retrieval_metadata: RetrievalMetadata
```

`retrieval_metadata` contiene:

```python
class RetrievalMetadata(BaseModel):
    question: str
    original_question: str | None
    translated_question: str | None
    status: str | None
    chunks_used: int
    chunks_metadata: list[ChunkMetadata]
```

Esto permite auditar:

- Que pregunto el usuario.
- Si se tradujo.
- Cuantos chunks se usaron.
- Que chunks se usaron.
- De donde salio cada fuente.
- Que veredicto de evidencia hubo.

## 20. Flujo completo en pseudocodigo

```python
def handle_user_query(question):
    pipeline = get_pipeline()  # cacheado
    return pipeline.run(question)
```

```python
def run(question):
    original_question = question

    is_spanish = detect_spanish(question)
    if is_spanish:
        question = translate_to_english(question)

    intent = route_query(question)

    if intent == "CHITCHAT":
        return chitchat_response(original_question)

    retrieved = hybrid_retrieve(question)
    reranked = flashrank_rerank(question, retrieved)
    evidence = check_evidence(question, reranked)

    if evidence.verdict == INSUFFICIENT:
        answer = abstention_message()
        if is_spanish:
            answer = translate_to_spanish(answer)
        return response_without_sources(answer, evidence)

    context = format_context(reranked)
    messages = build_messages(question, context)
    response = generate_with_groq(messages)

    if is_spanish:
        response.answer = translate_to_spanish(response.answer)

    response.retrieval_metadata.question = original_question
    response.retrieval_metadata.original_question = original_question
    return response
```

## 21. Ejemplo mental completo

Pregunta:

```text
Como persisto datos en un container Docker?
```

### 21.1. Idioma

```text
detect_spanish -> True
translate_to_english -> "How do I persist data in a Docker container?"
```

### 21.2. Router

La query contiene `Docker`, entonces:

```text
intent -> RAG
```

### 21.3. Retrieval semantico

Chroma busca chunks semanticamente parecidos a:

```text
How do I persist data in a Docker container?
```

Probablemente trae chunks sobre:

- Volumes.
- Bind mounts.
- Persisting container data.

### 21.4. BM25

BM25 busca terminos exactos:

```text
persist
data
Docker
container
```

### 21.5. RRF

Fusiona ambos rankings:

```text
score = semantic_weight/(60 + rank_semantic) + bm25_weight/(60 + rank_bm25)
```

### 21.6. FlashRank

Reevalua query + chunk juntos y deja los 10 mejores.

### 21.7. Evidence checker

Si el top chunk tiene buen `rerank_score` y hay suficientes chunks relevantes:

```text
verdict -> SUFFICIENT
```

### 21.8. Prompt

El contexto queda:

```text
[Source 1]
Document: Persisting Container Data
Section: ...
---
...
---

[Source 2]
Document: Volumes
Section: ...
---
...
---
```

### 21.9. Generacion

Groq genera algo en ingles con citas:

```text
To persist data for a Docker container, use volumes ... [Source 1]
```

### 21.10. Traduccion final

Como la pregunta original era en espanol, se traduce:

```text
Para persistir datos en un container Docker, usa volumes ... [Source 1]
```

Se preservan comandos, paths y terminos Docker.

## 22. Tecnicas aplicadas en la busqueda

Resumen de tecnicas:

| Tecnica | Donde | Para que sirve |
|---|---|---|
| Embeddings | `vector_store.py` | Representar significado semantico |
| Vector search | ChromaDB | Encontrar chunks similares por significado |
| BM25 | `bm25_retriever.py` | Encontrar terminos exactos, comandos y errores |
| Hybrid retrieval | `hybrid.py` | Combinar semantico + lexico |
| RRF | `hybrid.py` | Fusionar rankings sin normalizar scores |
| Cross-encoder reranking | `reranker.py` | Reordenar con mas precision query-documento |
| Evidence checking | `evidence_checker.py` | Evitar respuestas sin sustento |
| Grounded prompting | `prompt_templates.py` | Forzar al LLM a usar solo contexto |
| Citation parsing | `generator.py` | Vincular `[Source N]` con chunks reales |
| Traduccion hibrida | `translator.py` | Mejorar retrieval en corpus ingles y responder en espanol |

## 23. Puntos importantes para no confundirse

1. `pipeline.py` no hace la busqueda semantica directamente. La delega a `vector_store.py` y `hybrid.py`.

2. La busqueda semantica no compara texto contra texto. Compara vectores generados por `nomic-embed-text`.

3. BM25 no usa embeddings. Es busqueda lexica por terminos.

4. El score usado por `evidence_checker.py` es `rerank_score`, generado por FlashRank.

5. `LOW_CONFIDENCE` no detiene el pipeline. Solo `INSUFFICIENT` produce abstencion.

6. Las citas dependen de que el LLM escriba `[Source N]`. Si no las escribe, `sources` puede quedar vacio aunque haya usado contexto.

7. `SourceCitation.relevance_score` actualmente puede quedar en `0.0` porque el codigo guarda `rerank_score`, no `relevance_score`.

8. Si la pregunta es en espanol, retrieval y generacion ocurren internamente en ingles.

9. El pipeline se cachea con `@lru_cache`, asi que los componentes no se recrean en cada request.

10. BM25 se reconstruye desde Chroma al iniciar el pipeline; no se guarda en disco como indice independiente.

## 24. Diagrama final

```text
OpenWebUI / /query
        |
        v
FastAPI main.py
        |
        v
get_pipeline()  [cacheado]
        |
        v
RAGPipeline.run(question)
        |
        v
detect_spanish()
        |
        +-- espanol -> translate_to_english()
        |
        v
route_query()
        |
        +-- CHITCHAT -> get_chitchat_response() -> RAGResponse
        |
        +-- RAG
              |
              v
        semantic_retriever.invoke()
              |
              +-- query embedding con nomic-embed-text
              +-- ChromaDB top_k=30
              |
              v
        bm25_retriever.invoke()
              |
              v
        RRF fusion: semantico 0.4 + BM25 0.6
              |
              v
        FlashRank rerank top_n=10
              |
              v
        check_evidence()
              |
              +-- INSUFFICIENT -> abstention -> RAGResponse
              |
              +-- SUFFICIENT / LOW_CONFIDENCE
                      |
                      v
                format_context()
                      |
                      v
                build_messages()
                      |
                      v
                Groq llama-3.3-70b-versatile
                      |
                      v
                parse_citations()
                      |
                      v
                translate_to_spanish() si aplica
                      |
                      v
                RAGResponse final
```

