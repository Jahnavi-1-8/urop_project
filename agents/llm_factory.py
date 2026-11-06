"""LLM Factory for Agentic AI.

Instantiates available chat models from environment variables:
- GEMINI_API_KEY / GOOGLE_API_KEY -> ChatGoogleGenerativeAI (gemini-1.5-flash / gemini-2.0-flash / gemini-2.5-flash)
- OPENAI_API_KEY -> ChatOpenAI (gpt-4o-mini / gpt-3.5-turbo)
- GROQ_API_KEY -> ChatGroq
If no external API key is found, returns None, allowing deterministic grounded fallback.
"""

import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel


def get_configured_llm(temperature: float = 0.2) -> Optional[BaseChatModel]:
    """Retrieve an instantiated LangChain ChatModel based on available API keys."""
    # 1. Google Gemini
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Use standard gemini model name
            model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=gemini_key,
                temperature=temperature,
            )
        except Exception:
            pass

    # 2. OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            return ChatOpenAI(
                model=model_name,
                api_key=openai_key,
                temperature=temperature,
            )
        except Exception:
            pass

    return None

