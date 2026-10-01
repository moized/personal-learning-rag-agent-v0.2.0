"""Modern hybrid RAG retrieval implemented with zero extra runtime services.

Dense retrieval handles semantic similarity; BM25 handles exact technical terms;
RRF fuses both; an optional Gemini reranker chooses the final evidence set.
"""
from __future__ import annotations

import math
import re
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..models import Chunk, Document, Source, Group
from ..providers.base import AIProvider
from ..config import settings
from .grouping import cosine

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
        .where(Chunk.embedding_status == "done")
    )
    if group_ids:
        stmt = stmt.where(Chunk.group_id.in_(group_ids))
    if course_id is not None:
        stmt = stmt.where(Source.course_id == course_id)
    return db.execute(stmt.order_by(Chunk.id)).scalars().all()


def group_candidates(db: Session, query_embedding: list[float], primary_group_id: int | None, limit: int, course_id: int | None = None) -> list[int]:
    groups = db.execute(select(Group)).scalars().all()
    allowed: set[int] | None = None
    if course_id is not None:
        rows = db.execute(
            select(Chunk.group_id)
            .join(Document, Chunk.document_id == Document.id)
            .join(Source, Document.source_id == Source.id)
            .where(Source.course_id == course_id, Chunk.embedding_status == "done")
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


def _rank_dict(items: list[Chunk]) -> dict[int, int]:
    return {chunk.id: rank + 1 for rank, chunk in enumerate(items)}


def rrf_fuse(dense: list[tuple[Chunk, float]], lexical: list[tuple[Chunk, float]], k: int = 60) -> list[tuple[Chunk, float]]:
    by_id = {c.id: c for c, _ in dense}
    by_id.update({c.id: c for c, _ in lexical})
    scores: defaultdict[int, float] = defaultdict(float)
    for rank, (chunk, _) in enumerate(dense, 1):
        scores[chunk.id] += 1.0 / (k + rank)
    for rank, (chunk, _) in enumerate(lexical, 1):
        scores[chunk.id] += 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: settings.fused_candidate_k]
    return [(by_id[cid], score) for cid, score in ranked]


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


def _llm_rerank(ai: AIProvider, question: str, candidates: list[tuple[Chunk, float]], final_k: int) -> list[tuple[Chunk, float]]:
    if not settings.use_llm_reranker or not candidates:
        return candidates[:final_k]
    blocks = []
    for chunk, fused in candidates:
        blocks.append(f"CHUNK_ID={chunk.id}\nFUSED={fused:.6f}\nTEXT={(chunk.contextual_text or chunk.text)[:2800]}")
    prompt = f"""Rank the evidence for this learning question. Do not invent missing evidence.
Question: {question}
Candidates:
{chr(10).join(blocks)}
Return JSON with a `ranked` array. Each item must contain `chunk_id` and `relevance` from 0 to 1. Include only candidate IDs.
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
                ranked.append((lookup[cid], max(0.0, min(1.0, float(item.get("relevance", 0.0))))))
        if ranked:
            return ranked[:final_k]
    except Exception:
        pass
    return candidates[:final_k]



def expand_context_window(db: Session, ranked: list[tuple[Chunk, float]], radius: int = 1, max_items: int = 12) -> list[tuple[Chunk, float]]:
    """Add immediate same-document neighbors for lecture/transcript continuity."""
    if not ranked:
        return ranked
    seen = {c.id for c, _ in ranked}
    additions: list[tuple[Chunk, float]] = []
    for chunk, score in ranked:
        neighbors = db.execute(
            select(Chunk).where(
                Chunk.document_id == chunk.document_id,
                Chunk.chunk_index >= max(0, chunk.chunk_index - radius),
                Chunk.chunk_index <= chunk.chunk_index + radius,
                Chunk.embedding_status == "done",
            ).order_by(Chunk.chunk_index)
        ).scalars().all()
        for neighbor in neighbors:
            if neighbor.id in seen:
                continue
            seen.add(neighbor.id)
            additions.append((neighbor, score * 0.85))
            if len(ranked) + len(additions) >= max_items:
                break
        if len(ranked) + len(additions) >= max_items:
            break
    return (ranked + additions)[:max_items]

def retrieve(
    db: Session,
    ai: AIProvider,
    question: str,
    group_ids: list[int] | None = None,
    top_k: int | None = None,
    course_id: int | None = None,
) -> list[tuple[Chunk, float]]:
    top_k = top_k or settings.final_context_k
    query_embedding = ai.embed([question])[0]
    chunks = _course_chunks(db, course_id=course_id, group_ids=group_ids)
    dense = _dense_rank(query_embedding, chunks, settings.dense_candidate_k)
    lexical = _lexical_rank(question, chunks, settings.lexical_candidate_k)
    fused = rrf_fuse(dense, lexical)
    reranked = _llm_rerank(ai, question, fused, top_k)
    return expand_context_window(db, reranked, radius=1, max_items=min(top_k + 4, 12))


# Backward-compatible alias for older imports.
group_candidates_legacy = group_candidates
