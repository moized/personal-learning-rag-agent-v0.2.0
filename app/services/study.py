import json
import time
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Chunk, Document, Source, AgentTrace
from ..providers.base import AIProvider
from ..config import settings
from .knowledge_graph import resolve_concepts
from .retrieval import retrieve_many, group_candidates


def plan_query(ai: AIProvider, question: str) -> dict:
    """Produce a compact retrieval plan; the plan controls tools, not the final answer."""
    if not settings.use_llm_query_planner:
        return {
            "search_queries": [question],
            "intent": "study",
            "target_concepts": [],
            "needs_prerequisites": False,
            "prerequisite_depth": 0,
            "needs_source_comparison": False,
            "complexity": "low",
        }

    prompt = f"""Create a retrieval plan for this learning question.
Do not answer the question.
Preserve exact technical terminology and propose up to 3 complementary queries.
Identify likely canonical concept names that may exist in the learner knowledge graph.
Use prerequisites only when understanding the target plausibly depends on earlier concepts.

Question: {question}

Return JSON:
- search_queries: array of 1-3 strings
- intent: short label
- target_concepts: array of canonical concept names
- needs_prerequisites: boolean
- prerequisite_depth: integer 0-2
- needs_source_comparison: boolean
- complexity: low, medium, or high
"""
    schema = {
        "type": "object",
        "properties": {
            "search_queries": {"type": "array", "items": {"type": "string"}},
            "intent": {"type": "string"},
            "target_concepts": {"type": "array", "items": {"type": "string"}},
            "needs_prerequisites": {"type": "boolean"},
            "prerequisite_depth": {"type": "integer"},
            "needs_source_comparison": {"type": "boolean"},
            "complexity": {"type": "string"},
        },
        "required": [
            "search_queries",
            "intent",
            "target_concepts",
            "needs_prerequisites",
            "prerequisite_depth",
            "needs_source_comparison",
            "complexity",
        ],
    }
    try:
        result = ai.generate_json(prompt, task="routing", schema=schema)
        queries = [str(q).strip() for q in result.get("search_queries", []) if str(q).strip()]
        concepts = [str(c).strip() for c in result.get("target_concepts", []) if str(c).strip()]
        return {
            "search_queries": queries[:3] or [question],
            "intent": str(result.get("intent", "study")),
            "target_concepts": concepts[:8],
            "needs_prerequisites": bool(result.get("needs_prerequisites", False)),
            "prerequisite_depth": max(0, min(2, int(result.get("prerequisite_depth", 0)))),
            "needs_source_comparison": bool(result.get("needs_source_comparison", False)),
            "complexity": str(result.get("complexity", "medium")),
        }
    except Exception:
        return {
            "search_queries": [question],
            "intent": "study",
            "target_concepts": [],
            "needs_prerequisites": False,
            "prerequisite_depth": 0,
            "needs_source_comparison": False,
            "complexity": "medium",
        }


def answer(
    db: Session,
    ai: AIProvider,
    question: str,
    group_id: int | None = None,
    top_k: int = 8,
    course_id: int | None = None,
) -> dict:
    run_id = uuid4().hex
    started = time.perf_counter()
    query_plan = plan_query(ai, question)

    # Use one planning decision to drive group routing and the downstream retrieval plan.
    q = ai.embed([query_plan["search_queries"][0]])[0]
    selected_groups = group_candidates(
        db,
        q,
        group_id,
        limit=settings.retrieval_group_limit,
        course_id=course_id,
    )

    effective_top_k = min(20, top_k + (4 if query_plan["complexity"] == "high" else 0))
    primary = retrieve_many(
        db,
        ai,
        query_plan["search_queries"],
        group_ids=selected_groups,
        top_k=effective_top_k,
        course_id=course_id,
        target_concepts=query_plan["target_concepts"],
        prerequisite_depth=query_plan["prerequisite_depth"] if query_plan["needs_prerequisites"] else 0,
    )

    # A low-coverage result can escape routing restrictions once, preventing a bad group
    # classification from hiding useful evidence.
    if len(primary) < max(2, effective_top_k // 2):
        fallback = retrieve_many(
            db,
            ai,
            query_plan["search_queries"],
            group_ids=None,
            top_k=effective_top_k,
            course_id=course_id,
            target_concepts=query_plan["target_concepts"],
            prerequisite_depth=query_plan["prerequisite_depth"] if query_plan["needs_prerequisites"] else 0,
        )
        seen = {c.id for c, _ in primary}
        primary.extend((c, s) for c, s in fallback if c.id not in seen)
        primary = primary[:effective_top_k]

    graph_concepts = resolve_concepts(db, query_plan["target_concepts"])
    concept_names = [c.canonical_name for c in graph_concepts]

    context_blocks = []
    for chunk, score in primary:
        document = db.get(Document, chunk.document_id)
        source = db.get(Source, document.source_id) if document else None
        kind = "parent-window" if chunk.chunk_kind == "parent" else "evidence"
        context_blocks.append(
            f"[{kind} chunk:{chunk.id} source:{source.name if source else 'unknown'} "
            f"section:{document.section if document and document.section else ''} "
            f"score:{score:.4f}]\\n{chunk.text}"
        )
    context = "\\n\\n".join(context_blocks)

    prompt = f"""You are a rigorous study tutor.
Answer only from the supplied evidence.
Distinguish direct evidence from synthesis, and explicitly say when the evidence is insufficient.
Use concise citations like [chunk:123] next to factual claims when possible.
When prerequisite evidence is present, use it only to explain dependencies needed for the target topic.
Question: {question}
Evidence:
{context}
"""

    generation_started = time.perf_counter()
    answer_text = ai.generate_text(prompt, task="study")
    total_ms = int((time.perf_counter() - started) * 1000)
    generation_ms = int((time.perf_counter() - generation_started) * 1000)

    db.add(
        AgentTrace(
            run_id=run_id,
            event_type="study",
            task="study",
            model=settings.gemini_reasoning_model,
            status="ok",
            latency_ms=total_ms,
            payload_json=json.dumps(
                {
                    "question": question,
                    "query_plan": query_plan,
                    "selected_group_ids": selected_groups,
                    "resolved_target_concepts": concept_names,
                },
                ensure_ascii=False,
            ),
            result_json=json.dumps(
                {
                    "retrieved_chunk_ids": [c.id for c, _ in primary],
                    "evidence_count": len(primary),
                    "generation_ms": generation_ms,
                },
                ensure_ascii=False,
            ),
        )
    )
    db.flush()

    return {
        "answer": answer_text,
        "run_id": run_id,
        "sources": [{"chunk_id": c.id, "score": s, "kind": c.chunk_kind} for c, s in primary],
        "selected_group_ids": selected_groups,
        "query_plan": query_plan,
        "resolved_target_concepts": concept_names,
        "retrieval_architecture": (
            "adaptive: multi-query dense + BM25 → one RRF fusion → prerequisite graph expansion → "
            "one LLM rerank → parent/neighbor context"
        ),
    }
