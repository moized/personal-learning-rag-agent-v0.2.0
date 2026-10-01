import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class LocalProvider:
    """Offline deterministic fallback. It keeps the pipeline testable without an API key."""
    def __init__(self, dim: int = 768):
        self.vectorizer = HashingVectorizer(
            n_features=dim, alternate_sign=False, norm="l2", ngram_range=(1, 2), stop_words="english"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.vectorizer.transform(texts).toarray().astype(float).tolist()

    def generate_json(self, prompt: str, task: str = "extraction", schema: dict | None = None) -> dict:
        return {"group_name": "Uncategorized", "concepts": [], "relationships": [], "search_queries": []}

    def generate_text(self, prompt: str, task: str = "study") -> str:
        return "Local fallback is enabled. Configure GEMINI_API_KEY for generated lessons and planning."
