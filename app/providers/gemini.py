import json
import time
from typing import Any
import httpx


class GeminiProvider:
    """Gemini REST client with task-aware model routing and automatic fallback."""

    def __init__(
        self,
        api_key: str,
        reasoning_model: str,
        fast_model: str,
        fallback_model: str,
        embedding_model: str,
        retry_max: int = 4,
        reasoning_level: str = "medium",
        fast_thinking_level: str = "minimal",
    ):
        self.api_key = api_key
        self.reasoning_model = reasoning_model
        self.fast_model = fast_model
        self.fallback_model = fallback_model
        self.embedding_model = embedding_model
        self.retry_max = retry_max
        self.reasoning_level = reasoning_level
        self.fast_thinking_level = fast_thinking_level
        self.client = httpx.Client(timeout=180)

    def _post(self, url: str, payload: dict) -> dict:
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        last: Exception | None = None
        for attempt in range(self.retry_max + 1):
            try:
                response = self.client.post(url, headers=headers, json=payload)
                if response.status_code in (429, 500, 502, 503, 504):
                    last = RuntimeError(f"temporary Gemini error {response.status_code}: {response.text[:500]}")
                    if attempt < self.retry_max:
                        time.sleep(min(2 ** attempt, 30))
                        continue
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last = exc
                if attempt < self.retry_max:
                    time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(str(last))

    def _model_order(self, task: str) -> list[str]:
        # High-value synthesis/planning gets 3.8 first. Background extraction/routing stays cheap.
        if task in {"synthesis", "syllabus", "study", "agent_plan"}:
            primary = self.reasoning_model
        else:
            primary = self.fast_model
        return list(dict.fromkeys([primary, self.fallback_model, self.reasoning_model]))

    def _generate(self, prompt: str, model: str, *, json_schema: dict | None = None, thinking_level: str | None = None) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        config: dict[str, Any] = {}
        if thinking_level:
            config["thinkingConfig"] = {"thinkingLevel": thinking_level}
        if json_schema:
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = json_schema
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": config,
        }
        data = self._post(url, payload)
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {str(data)[:500]}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if not text.strip():
            raise RuntimeError("Gemini returned empty text")
        return text

    def generate_text(self, prompt: str, task: str = "study") -> str:
        errors = []
        for model in self._model_order(task):
            level = self.reasoning_level if model == self.reasoning_model else self.fast_thinking_level
            try:
                return self._generate(prompt, model, thinking_level=level)
            except Exception as exc:
                errors.append(f"{model}: {exc}")
        raise RuntimeError("All Gemini text models failed: " + " | ".join(errors))

    def generate_json(self, prompt: str, task: str = "extraction", schema: dict | None = None) -> dict:
        # A schema makes machine-readable agent decisions deterministic when the API supports it.
        effective_schema = schema or {
            "type": "object",
            "properties": {},
        }
        errors = []
        for model in self._model_order(task):
            level = self.reasoning_level if model == self.reasoning_model else self.fast_thinking_level
            try:
                text = self._generate(prompt, model, json_schema=effective_schema, thinking_level=level)
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    errors.append(f"{model}: invalid JSON: {exc}")
            except Exception as exc:
                errors.append(f"{model}: {exc}")
        raise RuntimeError("All Gemini JSON models failed: " + " | ".join(errors))

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.embedding_model}:batchEmbedContents"
        payload = {
            "requests": [
                {"model": f"models/{self.embedding_model}", "content": {"parts": [{"text": t}]}}
                for t in texts
            ]
        }
        data = self._post(url, payload)
        return [item["values"] for item in data.get("embeddings", [])]
