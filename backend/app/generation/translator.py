"""
Módulo para detectar lenguaje y traducir consultas y respuestas.

Estrategia híbrida de traducción (0-cost donde es posible):
  - Detección de idioma: `langdetect` (local, 0 tokens).
  - Query ES→EN: `deep-translator` / Google Translate (gratis, 0 tokens).
    La query del usuario es texto plano, no necesita inteligencia especial.
  - Respuesta EN→ES: LLM (Groq). La respuesta contiene bloques de código,
    comandos Docker y terminología técnica que Google Translate destruiría.
    Solo el LLM puede preservar `docker run -d -p 80:80 nginx` intacto.

Resultado: 1 llamada LLM por request en español (vs. 3 originales).
"""

from deep_translator import GoogleTranslator
from langchain_core.messages import SystemMessage, HumanMessage
from langdetect import detect, LangDetectException
from loguru import logger


def detect_spanish(llm, text: str) -> bool:
    """
    Detecta si el texto proporcionado está en español.
    Usa langdetect (determinístico, local, 0 tokens de API consumidos).

    El parámetro `llm` se mantiene por compatibilidad de firma,
    pero ya no se utiliza internamente.
    """
    try:
        lang = detect(text)
        is_spanish = lang == "es"
        logger.info(f"Detección de idioma (langdetect): '{lang}' → {'español' if is_spanish else 'otro'}")
        return is_spanish
    except LangDetectException as e:
        logger.warning(f"langdetect no pudo determinar el idioma: {e}. Asumiendo inglés.")
        return False


def translate_to_english(llm, text: str) -> str:
    """
    Traduce la query del usuario de español a inglés.
    Usa Google Translate (gratis, 0 tokens LLM).
    
    Es seguro usar Google Translate aquí porque la query del usuario
    es texto plano sin bloques de código ni comandos Docker.
    """
    try:
        logger.info("Traduciendo pregunta al inglés (Google Translate)...")
        translation = GoogleTranslator(source="es", target="en").translate(text)
        logger.info(f"Traducción completada: {translation}")
        return translation
    except Exception as e:
        logger.error(f"Error traduciendo a inglés: {e}")
        return text


def translate_to_spanish(llm, text: str) -> str:
    """
    Traduce la respuesta generada de inglés a español USANDO EL LLM.

    Se usa el LLM (no Google Translate) porque la respuesta contiene:
    - Bloques de código (docker run, docker compose up, etc.)
    - Terminología técnica (containers, volumes, images, etc.)
    - Paths de archivos y variables de entorno
    
    El LLM entiende el contexto y preserva todo esto intacto.
    Costo: 1 llamada LLM por request en español.
    """
    messages = [
        SystemMessage(content=(
            "You are a professional technical translator. Translate the following text "
            "from English to Spanish. CRITICAL RULES:\n"
            "1. NEVER translate code blocks, commands, file paths, or environment variables.\n"
            "2. Keep Docker terminology in English: containers, volumes, images, compose, "
            "Dockerfile, daemon, build, run, pull, push, etc.\n"
            "3. Keep product names in English: Docker, Windows, Linux, Ubuntu, Hyper-V, WSL, etc.\n"
            "4. Only translate the natural language explanatory text.\n"
            "5. Output ONLY the translation, absolutely nothing else."
        )),
        HumanMessage(content=text)
    ]
    try:
        logger.info("Traduciendo respuesta al español (LLM - preserva código)...")
        response = llm.invoke(messages)
        translation = response.content.strip()
        logger.info("Traducción de respuesta completada.")
        return translation
    except Exception as e:
        logger.error(f"Error traduciendo a español: {e}")
        return text
