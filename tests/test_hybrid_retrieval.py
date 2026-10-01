from app.services.retrieval import bm25_scores, rrf_fuse
from app.models import Chunk


def test_bm25_favors_exact_technical_term():
    chunks = [
        Chunk(id=1, document_id=1, chunk_index=0, text="Vector search retrieves semantic neighbors", contextual_text="Vector search retrieves semantic neighbors"),
        Chunk(id=2, document_id=1, chunk_index=1, text="BM25 matches exact technical terms", contextual_text="BM25 matches exact technical terms"),
    ]
    scores = bm25_scores("BM25 exact technical term", chunks)
    assert scores[1] > scores[0]


def test_rrf_merges_two_rankers_without_duplicates():
    c1 = Chunk(id=1, document_id=1, chunk_index=0, text="a")
    c2 = Chunk(id=2, document_id=1, chunk_index=1, text="b")
    dense = [(c1, 0.9), (c2, 0.8)]
    lexical = [(c2, 4.0), (c1, 1.0)]
    fused = rrf_fuse(dense, lexical)
    assert {c.id for c, _ in fused} == {1, 2}
