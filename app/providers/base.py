from typing import Protocol


class ProviderAuthenticationError(RuntimeError):
    """Raised when provider credentials are rejected and retrying cannot help."""


class ProviderRateLimitError(RuntimeError):
    """Raised when a provider continues returning HTTP 429 after backoff."""


class ProviderQuotaExceededError(ProviderRateLimitError):
    """Raised when Gemini reports a daily or unavailable quota, not a transient burst limit."""


class AIProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def generate_json(self, prompt: str, task: str = "extraction", schema: dict | None = None) -> dict: ...
    def generate_text(self, prompt: str, task: str = "study") -> str: ...
