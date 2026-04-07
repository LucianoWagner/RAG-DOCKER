"""
Módulo para detectar lenguaje y traducir consultas y respuestas.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger

def detect_spanish(llm, text: str) -> bool:
    """
    Detecta si el texto proporcionado está en español.
    Utiliza el LLM para responder con YES o NO.
    """
    messages = [
        SystemMessage(content="You are a language detection bot. Respond with exactly one word: 'YES' if the following text is in Spanish, or 'NO' otherwise."),
        HumanMessage(content=text)
    ]
    try:
        logger.info("Detectando si la pregunta está en español...")
        response = llm.invoke(messages)
        is_spanish = "YES" in response.content.strip().upper()
        logger.info(f"¿Es español? {'Sí' if is_spanish else 'No'} (Respuesta LLM: {response.content.strip()})")
        return is_spanish
    except Exception as e:
        logger.error(f"Error detectando idioma: {e}")
        return False

def translate_to_english(llm, text: str) -> str:
    """
    Traduce el texto dado de español a inglés.
    """
    messages = [
        SystemMessage(content="You are a professional technical translator. Translate the following text from Spanish to English. Only output the translation, absolutely nothing else."),
        HumanMessage(content=text)
    ]
    try:
        logger.info("Traduciendo pregunta al inglés...")
        response = llm.invoke(messages)
        translation = response.content.strip()
        logger.info(f"Traducción completada: {translation}")
        return translation
    except Exception as e:
        logger.error(f"Error traduciendo a inglés: {e}")
        return text

def translate_to_spanish(llm, text: str) -> str:
    """
    Traduce el texto dado de inglés a español técnico.
    """
    messages = [
        SystemMessage(content="You are a professional technical translator. Translate the following text from English to Spanish. Retain all technical docker terminology in English (like containers, volumes, compose, etc) if appropriate. Only output the translation, absolutely nothing else."),
        HumanMessage(content=text)
    ]
    try:
        logger.info("Traduciendo respuesta al español...")
        response = llm.invoke(messages)
        translation = response.content.strip()
        logger.info("Traducción de respuesta completada.")
        return translation
    except Exception as e:
        logger.error(f"Error traduciendo a español: {e}")
        return text
