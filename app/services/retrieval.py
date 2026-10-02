"""Adaptive hybrid retrieval with graph-aware context expansion."""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Chunk, Document, Source, Group
from ..providers.base import AIProvider
from .grouping import cosine
from .knowledge_graph import prerequisite_evidence_chunks

TOKEN_RE = re.compile(r"[\w./:+#-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if len(t) > 1]


def bm25_scores(query: str, chunks: list[Chunk]) -> list[float]:
    q = tokenize(query)
    if not q or not chunks:
        return [0.0] * len(chunks)
    docs = [tokenize(c.contextual_text or c.text) for c in chunks]
    n = len(docs)
    avgdl = sum(map(len, docs)) / max(n, 1)
    df: Counter[str] = Counter()
    for tokens in docs:
        for term in set(tokens):
            df[term] += 1
    k1, b = 1.5, 0.75
    scores: list[float] = []
    for tokens in docs:
        tf = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for term in q:
            if not tf[term]:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * dl / max(avgdl, 1.0))
            score += idf * (tf[term] * (k1 + 1)) / denom
        scores.append(score)
    return scores


def _course_chunks(db: Session, course_id: int | None, group_ids: list[int] | None = None) -> list[Chunk]:
    stmt = (
        select(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .join(Source, Document.source_id == Source.id)
        .where(Chunk.embedding_status == "done", Chunk.chunk_kind == "leaf")
    )
    if group_ids:
        stmt = stmt.where(Chunk.group_id.in_(group_ids))
    if course_id is not None:
        stmt = stmt.where(Source.course_id == course_id)
    return db.execute(stmt.order_by(Chunk.id)).scalars().all()


def group_candidates(
    db: Session,
    query_embedding: list[float],
    primary_group_id: int | None,
    limit: int,
    course_id: int | None = None,
) -> list[int]:
    groups = db.execute(select(Group)).scalars().all()
    allowed: set[int] | None = None
    if course_id is not None:
        rows = db.execute(
            select(Chunk.group_id)
            .join(Document, Chunk.document_id == Document.id)
            .join(Source, Document.source_id == Source.id)
            .where(
                Source.course_id == course_id,
                Chunk.embedding_status == "done",
                Chunk.chunk_kind == "leaf",
            )
        ).all()
        allowed = {row[0] for row in rows if row[0] is not None}
    scored: list[tuple[int, float]] = []
    for group in groups:
        if allowed is not None and group.id not in allowed:
            continue
        if not group.centroid_json:
            continue
        try:
            score = cosine(query_embedding, json.loads(group.centroid_json))
        except Exception:
            continue
        scored.append((group.id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    ids: list[int] = []
    if primary_group_id is not None and (allowed is None or primary_group_id in allowed):
        ids.append(primary_group_id)
    ids.extend(gid for gid, _ in scored if gid not in ids)
    return ids[:limit]


def _dense_rank(query_embedding: list[float], chunks: list[Chunk], limit: int) -> list[tuple[Chunk, float]]:
    scored: list[tuple[Chunk, float]] = []
    for chunk in chunks:
        if not chunk.embedding_json:
            continue
        try:
            score = cosine(query_embedding, json.loads(chunk.embedding_json))
        except Exception:
            continue
        scored.append((chunk, score))
    return sorted(scored, key=lambda x: x[1], reverse=True)[:limit]


def _lexical_rank(question: str, chunks: list[Chunk], limit: int) -> list[tuple[Chunk, float]]:
    values = bm25_scores(question, chunks)
    ranked = sorted(zip(chunks, values), key=lambda x: x[1], reverse=True)
    return [item for item in ranked[:limit] if item[1] > 0]


def rrf_fuse(
    dense: list[tuple[Chunk, float]],
    lexical: list[tuple[Chunk, float]],
    k: int = 60,
) -> list[tuple[Chunk, float]]:
    by_id = {c.id: c for c, _ in dense}
    by_id.update({c.id: c for c, _ in lexical})
    scores: defaultdict[int, float] = defaultdict(float)
    for rank, (chunk, _) in enumerate(dense, 1):
        scores[chunk.id] += 1.0 / (k + rank)
    for rank, (chunk, _) in enumerate(lexical, 1):
        scores[chunk.id] += 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(by_id[cid], score) for cid, score in ranked[: settings.fused_candidate_k]]


def rrf_fuse_many(
    retrieval_lists: list[tuple[list[tuple[Chunk, float]], list[tuple[Chunk, float]]]],
    k: int = 60,
) -> list[tuple[Chunk, float]]:
    """Fuse all query variants once, rather than reranking each query separately."""
    by_id: dict[int, Chunk] = {}
    scores: defaultdict[int, float] = defaultdict(float)
    for dense, lexical in retrieval_lists:
        for rank, (chunk, _) in enumerate(dense, 1):
            by_id[chunk.id] = chunk
            scores[chunk.id] += 1.0 / (k + rank)
        for rank, (chunk, _) in enumerate(lexical, 1):
            by_id[chunk.id] = chunk
            scores[chunk.id] += 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(by_id[cid], score) for cid, score in ranked[: settings.fused_candidate_k]]


def _llm_rerank(
    ai: AIProvider,
    question: str,
    candidates: list[tuple[Chunk, float]],
    final_k: int,
) -> list[tuple[Chunk, float]]:
    if not settings.use_llm_reranker or not candidates:
        return candidates[:final_k]
    blocks = []
    for chunk, fused in candidates:
        blocks.append(
            f"CHUNK_ID={chunk.id}\nFUSED={fused:.6f}\nTEXT={(chunk.contextual_text or chunk.text)[:2800]}"
        )
    prompt = f"""Rank the evidence for this learning question.
Do not invent missing evidence.
Give preference to evidence that directly answers the question, including prerequisite evidence when it explains a required missing concept.
Question: {question}
Candidates:
{chr(10).join(blocks)}
Return JSON with a ranked array. Each item must contain chunk_id and relevance from 0 to 1. Include only candidate IDs.
"""
    schema = {
        "type": "object",
        "properties": {
            "ranked": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "integer"},
                        "relevance": {"type": "number"},
                    },
                    "required": ["chunk_id", "relevance"],
                },
            }
        },
        "required": ["ranked"],
    }
    try:
        result = ai.generate_json(prompt, task="rerank", schema=schema)
        lookup = {c.id: c for c, _ in candidates}
        ranked: list[tuple[Chunk, float]] = []
        for item in result.get("ranked", []):
            cid = int(item.get("chunk_id", -1))
            if cid in lookup:
                ranked.append(
                    (
                        lookup[cid],
                        max(0.0, min(1.0, float(item.get("relevance", 0.0)))),
                    )
                )
        if ranked:
            return ranked[:final_k]
    except Exception:
        pass
    return candidates[:final_k]


def expand_context_window(
    db: Session,
    ranked: list[tuple[Chunk, float]],
    radius: int = 1,
    max_items: int = 12,
) -> list[tuple[Chunk, float]]:
    """Expand each selected leaf to its parent window and nearby leaves."""
    if not ranked:
        return ranked

    seen = {c.id for c, _ in ranked}
    additions: list[tuple[Chunk, float]] = []

    for chunk, score in ranked:
        if chunk.parent_chunk_id:
            parent = db.get(Chunk, chunk.parent_chunk_id)
            if parent and parent.id not in seen:
                seen.add(parent.id)
                additions.append((parent, score * 0.90))
                if len(ranked) + len(additions) >= max_items:
                    break

        neighbors = db.execute(
            select(Chunk).where(
                Chunk.document_id == chunk.document_id,
                Chunk.chunk_kind == "leaf",
                Chunk.chunk_index >= max(0, chunk.chunk_index - radius),
                Chunk.chunk_index <= chunk.chunk_index + radius,
                Chunk.embedding_status == "done",
            ).order_by(Chunk.chunk_index)
        ).scalars().all()
        for neighbor in neighbors:
            if neighbor.id in seen:
                continue
            seen.add(neighbor.id)
            additions.append((neighbor, score * 0.82))
            if len(ranked) + len(additions) >= max_items:
                break
        if len(ranked) + len(additions) >= max_items:
            break

    return (ranked + additions)[:max_items]


def retrieve_many(
    db: Session,
    ai: AIProvider,
    queries: list[str],
    group_ids: list[int] | None = None,
    top_k: int | None = None,
    course_id: int | None = None,
    target_concepts: list[str] | None = None,
    prerequisite_depth: int = 0,
    rerank_question: str | None = None,
) -> list[tuple[Chunk, float]]:
    """Retrieve from multiple query views, fuse once, graph-expand, then rerank once."""
    queries = [q.strip() for q in queries if q and q.strip()]
    if not queries:
        return []

    final_k = top_k or settings.final_context_k
    chunks = _course_chunks(db, course_id=course_id, group_ids=group_ids)
    embeddings = ai.embed(queries)
    if len(embeddings) != len(queries):
        raise RuntimeError("Embedding provider returned a mismatched query batch size")

    retrieval_lists: list[
        tuple[list[tuple[Chunk, float]], list[tuple[Chunk, float]]]
    ] = []
    for query, embedding in zip(queries, embeddings):
        retrieval_lists.append(
            (
                _dense_rank(embedding, chunks, settings.dense_candidate_k),
                _lexical_rank(query, chunks, settings.lexical_candidate_k),
            )
        )

    fused = rrf_fuse_many(retrieval_lists)

    graph_items = prerequisite_evidence_chunks(
        db,
        target_concept_names=target_concepts or [],
        depth=min(prerequisite_depth, settings.prerequisite_max_depth),
        course_id=course_id,
        limit=settings.prerequisite_evidence_k,
    )

    by_id = {chunk.id: (chunk, score) for chunk, score in fused}
    for chunk, score, _concept in graph_items:
        current = by_id.get(chunk.id)
        if current is None:
            by_id[chunk.id] = (chunk, score)
        else:
            by_id[chunk.id] = (chunk, max(current[1], score))

    candidates = sorted(by_id.values(), key=lambda item: item[1], reverse=True)
    candidates = candidates[: settings.fused_candidate_k]
    reranked = _llm_rerank(ai, rerank_question or queries[0], candidates, final_k)
    return expand_context_window(
        db,
        reranked,
        radius=1,
        max_items=min(final_k + 5, 16),
    )


def retrieve(
    db: Session,
    ai: AIProvider,
    question: str,
    group_ids: list[int] | None = None,
    top_k: int | None = None,
    course_id: int | None = None,
) -> list[tuple[Chunk, float]]:
    return retrieve_many(
        db,
        ai,
        [question],
        group_ids=group_ids,
        top_k=top_k,
        course_id=course_id,
    )


group_candidates_legacy = group_candidates
