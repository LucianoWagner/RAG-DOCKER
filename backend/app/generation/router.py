"""
Semantic Router — Clasificador de intenciones del usuario.

Responsabilidad:
- Determinar si una consulta requiere el pipeline RAG completo
  (búsqueda en documentación) o si es una interacción conversacional
  que puede resolverse directamente (saludos, agradecimientos, etc.).

Esto evita invocar el retrieval híbrido, FlashRank y el LLM generador
para preguntas triviales como "Hola" o "Gracias", ahorrando latencia
y recursos computacionales.

Estrategia:
  1. Primero evalúa con heurísticas locales (regex/keywords) → 0 tokens.
  2. Si no hay match claro, consulta al LLM con un prompt few-shot
     ultra-compacto que retorna solo "RAG" o "CHITCHAT".
"""

import re

from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger


# =====================================================================
# Heurísticas locales (0 tokens, ~0ms)
# =====================================================================

# Patrones que inequívocamente NO son preguntas técnicas.
# Se evalúan antes de gastar una llamada LLM.
_CHITCHAT_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*(hola|hello|hi|hey|buenas|buen\s?d[ií]a|buenos?\s?(d[ií]as|tardes|noches))\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(gracias|thanks|thank\s*you|thx|ty|muchas\s*gracias)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(chau|adi[oó]s|bye|nos\s*vemos|hasta\s*(luego|pronto|la\s*vista))\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(ok|dale|perfecto|genial|bien|entendido|listo|excelente|copado|joya)\s*[!.?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*[👋🙏😊👍🎉]+\s*$"),  # Solo emojis
    re.compile(r"^\s*(qui[eé]n\s+(sos|eres)|what\s+are\s+you|who\s+are\s+you|c[oó]mo\s+te\s+llam[aá]s)\s*[?!.]*\s*$", re.IGNORECASE),
]

# Palabras clave técnicas que garantizan que la consulta es para el RAG,
# incluso si también contiene saludos (ej: "Hola, cómo instalo Docker?").
_RAG_KEYWORDS: list[str] = [
    "docker", "container", "compose", "dockerfile", "image", "volume",
    "network", "build", "run", "pull", "push", "swarm", "daemon",
    "install", "instalar", "configurar", "configure", "error", "log",
    "port", "puerto", "deploy", "kubernetes", "k8s", "registry",
    "command", "comando", "cli", "terminal", "sudo", "wsl",
]


def _heuristic_classify(question: str) -> str | None:
    """
    Intenta clasificar la consulta usando reglas locales.

    Returns:
        "RAG", "CHITCHAT", o None si no hay certeza suficiente.
    """
    text = question.strip().lower()

    # Si contiene keywords técnicas → RAG inmediato (aunque diga "Hola")
    if any(kw in text for kw in _RAG_KEYWORDS):
        return "RAG"

    # Si matchea un patrón de cháchara pura → CHITCHAT
    for pattern in _CHITCHAT_PATTERNS:
        if pattern.match(question):
            return "CHITCHAT"

    return None


# =====================================================================
# Clasificación LLM (fallback, ~0.3s en Groq)
# =====================================================================

_ROUTER_SYSTEM_PROMPT = """\
You are a query classifier for a Docker documentation assistant.
Your ONLY job is to decide if a user message requires searching documentation (RAG) \
or is just casual conversation (CHITCHAT).

Rules:
- RAG: Any question about Docker, containers, installation, configuration, \
commands, troubleshooting, or technical topics.
- CHITCHAT: Greetings, thanks, farewells, personal questions, jokes, \
or anything unrelated to Docker/technical documentation.

Respond with EXACTLY one word: RAG or CHITCHAT. Nothing else."""

_FEW_SHOT_EXAMPLES = [
    ("How do I install Docker on Ubuntu?", "RAG"),
    ("Hola, ¿cómo estás?", "CHITCHAT"),
    ("What is the difference between CMD and ENTRYPOINT?", "RAG"),
    ("Gracias por la ayuda!", "CHITCHAT"),
    ("My container keeps crashing with exit code 137", "RAG"),
    ("¿Quién sos?", "CHITCHAT"),
    ("Cómo expongo un puerto en Docker Compose?", "RAG"),
    ("jaja buenísimo", "CHITCHAT"),
]


def _llm_classify(llm, question: str) -> str:
    """
    Clasifica la consulta usando el LLM como fallback.

    Usa un prompt few-shot mínimo para que el modelo responda
    estrictamente con "RAG" o "CHITCHAT".
    """
    # Construir los ejemplos few-shot como parte del prompt
    examples_text = "\n".join(
        f'User: "{q}" → {label}' for q, label in _FEW_SHOT_EXAMPLES
    )

    messages = [
        SystemMessage(content=f"{_ROUTER_SYSTEM_PROMPT}\n\nExamples:\n{examples_text}"),
        HumanMessage(content=question),
    ]

    try:
        response = llm.invoke(messages)
        result = response.content.strip().upper()

        # Sanitizar: solo aceptar RAG o CHITCHAT
        if "RAG" in result:
            return "RAG"
        elif "CHITCHAT" in result or "CHIT" in result:
            return "CHITCHAT"
        else:
            # Si el LLM devuelve algo inesperado, defaultear a RAG
            # (es mejor buscar de más que ignorar una pregunta técnica)
            logger.warning(f"Router LLM devolvió respuesta inesperada: '{result}'. Defaulteando a RAG.")
            return "RAG"

    except Exception as e:
        logger.error(f"Error en router LLM: {e}. Defaulteando a RAG.")
        return "RAG"


# =====================================================================
# API Pública
# =====================================================================

def route_query(llm, question: str) -> str:
    """
    Clasifica la intención del usuario.

    Estrategia en cascada:
    1. Heurísticas locales (regex + keywords) → 0 tokens, ~0ms.
    2. Si no hay certeza, fallback al LLM → ~0.3s en Groq.

    Args:
        llm: Instancia del LLM (ChatGroq o ChatOllama).
        question: Pregunta del usuario en lenguaje natural.

    Returns:
        "RAG" si la consulta requiere búsqueda en documentación.
        "CHITCHAT" si es una interacción conversacional.
    """
    # 1. Intentar clasificación local (gratis)
    heuristic_result = _heuristic_classify(question)
    if heuristic_result is not None:
        logger.info(f"🚦 Router [heurística]: {heuristic_result} | '{question[:60]}'")
        return heuristic_result

    # 2. Fallback al LLM
    llm_result = _llm_classify(llm, question)
    logger.info(f"🚦 Router [LLM]: {llm_result} | '{question[:60]}'")
    return llm_result


def get_chitchat_response(llm, question: str) -> str:
    """
    Genera una respuesta conversacional amigable para queries chit-chat.

    Usa el LLM para dar una respuesta natural y contextualizada,
    en lugar de un mensaje fijo.

    Args:
        llm: Instancia del LLM.
        question: Mensaje conversacional del usuario.

    Returns:
        Respuesta amigable del asistente.
    """
    messages = [
        SystemMessage(content=(
            "You are a friendly Docker documentation assistant called 'Docker RAG Assistant'. "
            "The user sent a casual/conversational message (not a technical question). "
            "Respond warmly and briefly. Remind them you can help with Docker documentation "
            "if they need it. Keep it to 1-2 sentences max. "
            "Respond in the SAME LANGUAGE the user is using."
        )),
        HumanMessage(content=question),
    ]

    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error generando respuesta chit-chat: {e}")
        return (
            "¡Hola! 👋 Soy tu asistente de documentación Docker. "
            "Preguntame lo que necesites sobre Docker y te ayudo."
        )
