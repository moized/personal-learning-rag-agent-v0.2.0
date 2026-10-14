from typing import Protocol


class AIProvider(Protocol):
    # gemini: calls the API; manual: asks the user via Decision rows; local: offline fallbacks.
    mode: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def generate_json(self, prompt: str, task: str = "extraction", schema: dict | None = None, *,
                      key: str | None = None, label: str | None = None) -> dict: ...
    def generate_text(self, prompt: str, task: str = "study", *,
                      key: str | None = None, label: str | None = None) -> str: ...


class DecisionPending(Exception):
    """Raised in manual mode when a model decision has no pasted answer yet.

    The caller records it as a Decision row once its own transaction has rolled back,
    then pauses the job (or returns the prompt to the API client).
    """

    def __init__(self, key: str, task: str, kind: str, prompt: str, schema: dict | None, label: str | None):
        super().__init__(f"Waiting for an external AI answer: {label or task}")
        self.key = key
        self.task = task
        self.kind = kind
        self.prompt = prompt
        self.schema = schema
        self.label = label


def is_automatic(ai) -> bool:
    """Optional enrichment calls (profiles, reranking, query planning) only run against a real API."""
    return getattr(ai, "mode", "gemini") == "gemini"
