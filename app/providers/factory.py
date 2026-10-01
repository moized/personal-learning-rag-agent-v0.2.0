from .gemini import GeminiProvider
from .local import LocalProvider
from ..config import settings


def provider():
    if settings.gemini_api_key:
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            reasoning_model=settings.gemini_reasoning_model,
            fast_model=settings.gemini_fast_model,
            fallback_model=settings.gemini_fallback_model,
            embedding_model=settings.gemini_embedding_model,
            retry_max=settings.retry_max,
            reasoning_level=settings.gemini_reasoning_level,
            fast_thinking_level=settings.gemini_fast_thinking_level,
        )
    return LocalProvider(settings.embedding_dim)
