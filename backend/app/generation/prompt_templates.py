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

SYSTEM_PROMPT = """Eres un asistente técnico especializado en Docker. Tu función es responder
preguntas sobre instalación, uso inicial y resolución de errores de Docker,
basándote EXCLUSIVAMENTE en la documentación oficial proporcionada como contexto.

REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE con información presente en los fragmentos de contexto proporcionados.
2. Si el contexto no contiene información suficiente para responder, indícalo
   explícitamente. NUNCA inventes información ni uses conocimiento externo.
3. Cuando incluyas comandos, muéstralos en bloques de código con la sintaxis correcta.
4. Cita las fuentes usando el formato [Fuente N] al final de cada afirmación clave,
   donde N corresponde al número del fragmento de contexto utilizado.
5. Si la pregunta involucra una plataforma específica (Windows, Mac, Linux),
   enfoca tu respuesta en esa plataforma.
6. RESPONDE EN EL MISMO IDIOMA EN QUE SE FORMULA LA PREGUNTA.
7. Estructura tus respuestas de forma clara:
   - Para instalación: pasos numerados con comandos
   - Para troubleshooting: diagnóstico → causa probable → solución
   - Para conceptos: definición → explicación → ejemplo práctico

FRAGMENTOS DE DOCUMENTACIÓN (CONTEXTO):
{context}
"""

# =============================================================================
# User Prompt Template
# =============================================================================

USER_PROMPT = """PREGUNTA: {question}

Respondé basándote exclusivamente en los fragmentos de documentación proporcionados.
Incluí citas [Fuente N] para cada afirmación clave."""


# =============================================================================
# Variantes de Prompt (para evaluación comparativa)
# =============================================================================

# Prompt más restrictivo: enfatiza abstención sobre alucinación
SYSTEM_PROMPT_RESTRICTIVE = """Eres un asistente técnico de Docker que SOLO responde con información
del contexto proporcionado. Si la información no está en el contexto, DEBES decir
"No tengo información suficiente para responder esta pregunta."

NO USES conocimiento externo bajo ninguna circunstancia.
RESPONDE EN EL MISMO IDIOMA DE LA PREGUNTA.

CONTEXTO:
{context}
"""

# Prompt más permisivo: permite algo de razonamiento
SYSTEM_PROMPT_PERMISSIVE = """Eres un asistente técnico de Docker. Usá los fragmentos de
documentación proporcionados como base principal para tu respuesta.
Podés hacer inferencias razonables basadas en el contexto, pero priorizá
siempre la información explícita del contexto.

Citá las fuentes cuando sea posible usando [Fuente N].
RESPONDE EN EL MISMO IDIOMA DE LA PREGUNTA.

CONTEXTO:
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
        title = meta.get("doc_title", "Documentación")
        section = meta.get("section_header", "General")
        
        chunk_str = (
            f"[Fuente {n}]\n"
            f"Documento: {title}\n"
            f"Sección: {section}\n"
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
