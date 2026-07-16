"""Single place that constructs the chat model.

Using Groq's LLaMA-3.3-70B for fast, cheap inference (consistent with
the rest of your projects). Swapping providers later only means editing
this file.
"""

from langchain_groq import ChatGroq

from agentic_assistant.config import settings


def get_llm(temperature: float | None = None) -> ChatGroq:
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        temperature=temperature if temperature is not None else settings.temperature,
    )
