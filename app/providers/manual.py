import hashlib
import json
from typing import Callable
from sqlalchemy import select
from .base import DecisionPending


def decision_key(task: str, prompt: str, key: str | None = None) -> str:
    """Stable identity for a decision. Callers pass `key` when the prompt may drift between runs."""
    return f"{task}:{key}" if key else f"{task}:{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:24]}"


def schema_example(schema: dict | None):
    """Turn a JSON schema into an example object an external chat AI can imitate."""
    schema = schema or {}
    kind = schema.get("type")
    if kind == "object":
        return {name: schema_example(sub) for name, sub in (schema.get("properties") or {}).items()}
    if kind == "array":
        return [schema_example(schema.get("items"))]
    if kind == "integer":
        return 0
    if kind == "number":
        return 0.0
    if kind == "boolean":
        return False
    return schema.get("description") or "..."


def external_prompt(prompt: str, schema: dict | None) -> str:
    """The text the user copies into ChatGPT/Gemini/Claude for a JSON decision."""
    shape = json.dumps(schema_example(schema or {"type": "object", "properties": {}}), indent=2, ensure_ascii=False)
    return (
        f"{prompt.rstrip()}\n\n"
        "Reply with only a JSON object in exactly this shape (no commentary before or after it):\n"
        f"{shape}"
    )


class ManualProvider:
    """Human-in-the-loop provider: decisions are answered by the user with any external AI.

    Embeddings still run locally (or through Gemini when a key is configured) because they are
    numeric and cannot be pasted. Every generate_* call returns the stored answer when the user
    has pasted one, and otherwise raises DecisionPending.
    """
    mode = "manual"

    def __init__(self, embedder: Callable[[list[str]], list[list[float]]], session_factory=None):
        self._embed = embedder
        if session_factory is None:
            from ..db import SessionLocal
            session_factory = SessionLocal
        self.session_factory = session_factory

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def _stored(self, key: str):
        from ..models import Decision
        with self.session_factory() as db:
            decision = db.execute(select(Decision).where(Decision.key == key)).scalar_one_or_none()
            if decision and decision.status == "answered":
                return decision.result_json if decision.kind == "json" else decision.answer
        return None

    def generate_json(self, prompt: str, task: str = "extraction", schema: dict | None = None, *,
                      key: str | None = None, label: str | None = None) -> dict:
        k = decision_key(task, prompt, key)
        stored = self._stored(k)
        if stored is not None:
            return json.loads(stored)
        raise DecisionPending(k, task, "json", external_prompt(prompt, schema), schema, label)

    def generate_text(self, prompt: str, task: str = "study", *,
                      key: str | None = None, label: str | None = None) -> str:
        k = decision_key(task, prompt, key)
        stored = self._stored(k)
        if stored is not None:
            return stored
        raise DecisionPending(k, task, "text", prompt, None, label)
