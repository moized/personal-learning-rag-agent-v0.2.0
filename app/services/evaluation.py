"""Small, dependency-free retrieval evaluation helpers."""
from __future__ import annotations

from typing import Iterable


def recall_at_k(retrieved_ids: Iterable[int], relevant_ids: Iterable[int], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top = list(retrieved_ids)[:k]
    return len(set(top) & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids: Iterable[int], relevant_ids: Iterable[int]) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    for rank, item_id in enumerate(retrieved_ids, 1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(cases: list[dict], k_values: tuple[int, ...] = (5, 10)) -> dict:
    if not cases:
        return {"cases": 0, "mrr": 0.0, **{f"recall@{k}": 0.0 for k in k_values}}

    result = {
        "cases": len(cases),
        "mrr": sum(
            reciprocal_rank(case.get("retrieved_ids", []), case.get("relevant_ids", []))
            for case in cases
        ) / len(cases),
    }
    for k in k_values:
        result[f"recall@{k}"] = sum(
            recall_at_k(case.get("retrieved_ids", []), case.get("relevant_ids", []), k)
            for case in cases
        ) / len(cases)
    return result
