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
