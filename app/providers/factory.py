from .gemini import GeminiProvider
from .local import LocalProvider
from .manual import ManualProvider
from ..config import settings


def ai_mode() -> str:
    mode = (settings.ai_mode or "auto").strip().lower()
    if mode == "auto" or (mode == "gemini" and not settings.gemini_api_key):
        return "gemini" if settings.gemini_api_key else "manual"
    return mode if mode in {"manual", "local"} else "manual"


def _gemini() -> GeminiProvider:
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


def provider():
    mode = ai_mode()
    if mode == "gemini":
        return _gemini()
    if mode == "manual":
        # Decisions go to the user; embeddings use Gemini when a key exists, else local hashing.
        embedder = _gemini().embed if settings.gemini_api_key else LocalProvider(settings.embedding_dim).embed
        return ManualProvider(embedder)
    return LocalProvider(settings.embedding_dim)
