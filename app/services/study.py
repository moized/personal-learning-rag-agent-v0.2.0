import json
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..models import Chunk, Document, Source
from ..providers.base import AIProvider
from ..config import settings
from .retrieval import retrieve, group_candidates


def plan_query(ai: AIProvider, question: str) -> dict:
    if not settings.use_llm_query_planner:
        return {"search_queries": [question], "intent": "study"}
    prompt = f"""Turn this learning question into 1-3 retrieval queries that preserve exact technical terminology.
Do not answer the question. Avoid generic rewrites.
Question: {question}
Return JSON with `search_queries` and `intent`.
"""
    schema = {
        "type": "object",
        "properties": {
            "search_queries": {"type": "array", "items": {"type": "string"}},
            "intent": {"type": "string"},
        },
        "required": ["search_queries", "intent"],
    }
    try:
        result = ai.generate_json(prompt, task="routing", schema=schema)
        queries = [str(q).strip() for q in result.get("search_queries", []) if str(q).strip()]
        return {"search_queries": queries[:3] or [question], "intent": result.get("intent", "study")}
    except Exception:
        return {"search_queries": [question], "intent": "study"}


def retrieve_hybrid(db: Session, ai: AIProvider, question: str, group_ids: list[int] | None, top_k: int, course_id: int | None):
    plan = plan_query(ai, question)
    merged: dict[int, tuple[Chunk, float]] = {}
    for query in plan["search_queries"]:
        for chunk, score in retrieve(db, ai, query, group_ids, top_k, course_id):
            if chunk.id not in merged or score > merged[chunk.id][1]:
                merged[chunk.id] = (chunk, score)
    ranked = sorted(merged.values(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k], plan


def answer(db: Session, ai: AIProvider, question: str, group_id: int | None = None, top_k: int = 8, course_id: int | None = None) -> dict:
    q = ai.embed([question])[0]
    selected_groups = group_candidates(db, q, group_id, limit=settings.retrieval_group_limit, course_id=course_id)
    primary, query_plan = retrieve_hybrid(db, ai, question, selected_groups, top_k, course_id)
    if len(primary) < max(2, top_k // 2):
        fallback, _ = retrieve_hybrid(db, ai, question, None, top_k, course_id)
        seen = {c.id for c, _ in primary}
        primary.extend((c, s) for c, s in fallback if c.id not in seen)
        primary = primary[:top_k]

    context = "\n\n".join(
        f"[chunk:{c.id} source:{(db.get(Source, db.get(Document, c.document_id).source_id).name)} section:{db.get(Document, c.document_id).section or ''}]\n{c.text}"
        for c, _ in primary
    )
    prompt = f"""You are a rigorous study tutor.
Answer only from the supplied evidence. Distinguish direct evidence from synthesis, and explicitly say when the evidence is insufficient.
Use concise citations like [chunk:123] next to claims when possible.
Question: {question}
Evidence:
{context}
"""
    return {
        "answer": ai.generate_text(prompt, task="study"),
        "sources": [{"chunk_id": c.id, "score": s} for c, s in primary],
        "selected_group_ids": selected_groups,
        "query_plan": query_plan,
        "retrieval_architecture": "dense + BM25 + RRF + optional Gemini rerank",
    }
