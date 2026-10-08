from app.providers.local import LocalProvider


def test_local_embeddings():
    p = LocalProvider(dim=32)
    vecs = p.embed(["gradient descent optimization", "gradient descent optimization"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 32
    assert vecs[0] == vecs[1]


def test_gemini_task_routing_order():
    from app.providers.gemini import GeminiProvider
    p = GeminiProvider('x', 'gemini-3.8-flash', 'gemini-3.5-flash-lite', 'gemini-3.5-flash-lite', 'gemini-embedding-2')
    assert p._model_order('study')[0] == 'gemini-3.8-flash'
    assert p._model_order('extraction')[0] == 'gemini-3.5-flash-lite'
import pytest


def test_gemini_authentication_error_is_not_retried():
    from types import SimpleNamespace
    from app.providers.base import ProviderAuthenticationError
    from app.providers.gemini import GeminiProvider

    class UnauthorizedClient:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            return SimpleNamespace(status_code=401, text="invalid credential")

    provider = GeminiProvider("invalid", "reason", "fast", "fallback", "embedding", retry_max=4)
    client = UnauthorizedClient()
    provider.client = client
    with pytest.raises(ProviderAuthenticationError, match=r"authentication failed \(401\)"):
        provider.embed(["text"])
    assert client.calls == 1


def test_gemini_429_honors_retry_after(monkeypatch):
    from types import SimpleNamespace
    from app.providers.gemini import GeminiProvider
    import app.providers.gemini as gemini_module

    waits=[]
    monkeypatch.setattr(gemini_module.time, "sleep", waits.append)
    class Client:
        calls=0
        def post(self, *args, **kwargs):
            self.calls+=1
            if self.calls==1:
                return SimpleNamespace(status_code=429, text="quota", headers={"Retry-After":"7"})
            return SimpleNamespace(status_code=200, json=lambda:{"ok":True}, raise_for_status=lambda:None)
    provider=GeminiProvider("key", "reason", "fast", "fallback", "embedding", retry_max=1)
    provider.client=Client()
    assert provider._post("url", {}) == {"ok":True}
    assert waits == [7.0]


def test_gemini_exhausted_429_has_rate_limit_error():
    from types import SimpleNamespace
    from app.providers.base import ProviderRateLimitError
    from app.providers.gemini import GeminiProvider

    class Client:
        def post(self, *args, **kwargs):
            return SimpleNamespace(status_code=429, text="quota", headers={})
    provider=GeminiProvider("key", "reason", "fast", "fallback", "embedding", retry_max=0)
    provider.client=Client()
    with pytest.raises(ProviderRateLimitError):
        provider._post("url", {})
