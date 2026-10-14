import os
import tempfile
from pathlib import Path

# Point the app at a throwaway database before any app module creates its engine,
# and keep a developer's .env API key from turning tests into Gemini calls.
_TMP = Path(tempfile.mkdtemp(prefix="learning-agent-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"
os.environ["UPLOAD_DIR"] = str(_TMP / "uploads")
os.environ["GEMINI_API_KEY"] = ""
os.environ["AI_MODE"] = "manual"

import pytest  # noqa: E402


@pytest.fixture
def db():
    from app.db import Base, engine, init_db, SessionLocal
    Base.metadata.drop_all(engine)
    init_db()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


class ScriptedProvider:
    """Behaves like the Gemini provider but answers from canned functions keyed by task."""
    mode = "gemini"

    def __init__(self, **responses):
        from app.providers.local import LocalProvider
        self.local = LocalProvider(64)
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def embed(self, texts):
        return self.local.embed(texts)

    def generate_json(self, prompt, task="extraction", schema=None, *, key=None, label=None):
        self.calls.append((task, prompt))
        return self.responses[task](prompt)

    def generate_text(self, prompt, task="study", *, key=None, label=None):
        self.calls.append((task, prompt))
        return "generated"


@pytest.fixture
def scripted():
    return ScriptedProvider
