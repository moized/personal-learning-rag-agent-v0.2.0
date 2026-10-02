from app.services.evaluation import evaluate_retrieval, recall_at_k, reciprocal_rank


def test_recall_and_mrr():
    assert recall_at_k([1, 2, 3], [3], 3) == 1.0
    assert recall_at_k([1, 2, 3], [3], 2) == 0.0
    assert reciprocal_rank([5, 7, 9], [7]) == 0.5


def test_evaluate_retrieval():
    result = evaluate_retrieval([
        {"retrieved_ids": [1, 2, 3], "relevant_ids": [2]},
        {"retrieved_ids": [8, 4], "relevant_ids": [9]},
    ])
    assert result["cases"] == 2
    assert result["mrr"] == 0.25
    assert result["recall@5"] == 0.5
