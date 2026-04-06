"""
Prompt Templates — Prompts especializados para soporte técnico Docker.

Responsabilidad:
- Definir el system prompt principal del asistente
- Definir variantes de prompt para evaluación comparativa
- Formatear el contexto de chunks para insertar en el prompt

Decisiones de diseño:
- El prompt indica al LLM que responda en el IDIOMA DE LA PREGUNTA
- Se requieren citas inline [Fuente N] en la respuesta
- Se prohíbe inventar información fuera del contexto
- Se estructura la respuesta según el tipo de consulta
"""

# =============================================================================
# System Prompt Principal
# =============================================================================

SYSTEM_PROMPT = """You are a strictly grounded technical assistant specialized in Docker. Your sole purpose is to answer questions using ONLY the provided documentation context.

ANTI-HALLUCINATION PROTOCOL (CRITICAL):
1. You are FORBIDDEN from using any external or pre-trained knowledge.
2. If a specific detail (like a command, file path, socket location, version, or config option) is not EXPLICITLY written in the provided fragments, DO NOT INVENT IT. State clearly: "The provided documentation does not detail this."
3. Every single assertion you make MUST be directly verifiable against the context snippets.
4. If the context snippets do not address the user's problem at all, reply EXACTLY with: "I'm sorry, but the provided documentation does not contain the answer to your question."
5. Cite your sources using the format [Source N] at the end of every sentence or technical claim you make.
6. YOUR FINAL RESPONSE MUST ALWAYS BE IN ENGLISH, regardless of the language the user uses.

DOCUMENTATION FRAGMENTS (CONTEXT):
{context}
"""

# =============================================================================
# User Prompt Template
# =============================================================================

USER_PROMPT = """QUESTION: {question}

Answer based exclusively on the provided documentation fragments.
Include citations [Source N] for every key statement. All output must be in English."""


# =============================================================================
# Variantes de Prompt (para evaluación comparativa)
# =============================================================================

# Prompt más restrictivo: enfatiza abstención sobre alucinación
SYSTEM_PROMPT_RESTRICTIVE = """You are a Docker technical assistant that ONLY answers using information from the provided context. If the information is not in the context, you MUST say "I don't have enough information to answer this question."

DO NOT use external knowledge under any circumstances.
RESPOND IN ENGLISH.

CONTEXT:
{context}
"""

# Prompt más permisivo: permite algo de razonamiento
SYSTEM_PROMPT_PERMISSIVE = """You are a Docker technical assistant. Use the provided documentation fragments as the primary basis for your answer. You may make reasonable inferences based on the context, but always prioritize explicit information from the context.

Cite sources when possible using [Source N].
RESPOND IN ENGLISH.

CONTEXT:
{context}
"""


def format_context(chunks: list) -> str:
    """
    Formatea los chunks recuperados como texto de contexto para el prompt.

    Args:
        chunks: Lista de Documents con metadata.

    Returns:
        Texto formateado con los fragmentos numerados.
    """
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        # LangChain LLMs / Generadores asumen índice desde 1 para las fuentes (1-indexed para humanos)
        n = i + 1
        meta = chunk.metadata
        title = meta.get("doc_title", "Documentation")
        section = meta.get("section_header", "General")
        
        chunk_str = (
            f"[Source {n}]\n"
            f"Document: {title}\n"
            f"Section: {section}\n"
            f"---\n"
            f"{chunk.page_content}\n"
            f"---"
        )
        formatted_chunks.append(chunk_str)
        
    return "\n\n".join(formatted_chunks)


def build_messages(question: str, context: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    """
    Construye la lista de mensajes para enviar al LLM.

    Args:
        question: Pregunta del usuario.
        context: Texto de contexto formateado.
        system_prompt: Template del system prompt a usar.

    Returns:
        Lista de dicts con role/content para el LLM.
    """
    return [
        {
            "role": "system", 
            "content": system_prompt.format(context=context)
        },
        {
            "role": "user", 
            "content": USER_PROMPT.format(question=question)
        }
    ]
