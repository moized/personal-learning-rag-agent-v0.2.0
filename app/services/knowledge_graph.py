"""Knowledge graph primitives for dependency-aware retrieval and learning navigation."""
from __future__ import annotations

import json
from collections import deque

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Chunk, ChunkRelation, Concept, ConceptRelation, Evidence, Document, Source
from ..providers.base import AIProvider
from ..config import settings


RELATION_TYPES = {"prerequisite", "related", "contradicts"}


def add_sequence_relations(db: Session, document_id: int) -> int:
    leaves = db.execute(
        select(Chunk)
        .where(Chunk.document_id == document_id, Chunk.chunk_kind == "leaf")
        .order_by(Chunk.chunk_index)
    ).scalars().all()
    created = 0
    for current, nxt in zip(leaves, leaves[1:]):
        exists = db.execute(
            select(ChunkRelation).where(
                ChunkRelation.from_chunk_id == current.id,
                ChunkRelation.to_chunk_id == nxt.id,
                ChunkRelation.relation_type == "next",
            )
        ).scalar_one_or_none()
        if not exists:
            db.add(
                ChunkRelation(
                    from_chunk_id=current.id,
                    to_chunk_id=nxt.id,
                    relation_type="next",
                    confidence=1.0,
                )
            )
            created += 1
    return created


def resolve_concepts(db: Session, names: list[str]) -> list[Concept]:
    if not names:
        return []
    normalized = {str(n).strip().lower() for n in names if str(n).strip()}
    concepts = db.execute(select(Concept)).scalars().all()
    matched: list[Concept] = []
    for concept in concepts:
        candidates = [concept.canonical_name]
        try:
            aliases = json.loads(concept.aliases_json or "[]")
            if isinstance(aliases, list):
                candidates.extend(str(a) for a in aliases)
        except json.JSONDecodeError:
            pass
        candidate_norm = {c.strip().lower() for c in candidates if c and c.strip()}
        if candidate_norm & normalized:
            matched.append(concept)
            continue
        for requested in normalized:
            if any(requested in c or c in requested for c in candidate_norm if c):
                matched.append(concept)
                break
    return matched


def prerequisite_chain(
    db: Session,
    target_concept_ids: list[int],
    depth: int = 2,
    course_id: int | None = None,
) -> list[tuple[Concept, int]]:
    if not target_concept_ids or depth <= 0:
        return []

    queue = deque((cid, 0) for cid in target_concept_ids)
    visited = set(target_concept_ids)
    results: list[tuple[Concept, int]] = []

    while queue:
        dependent_id, current_depth = queue.popleft()
        if current_depth >= depth:
            continue

        relations = db.execute(
            select(ConceptRelation).where(
                ConceptRelation.to_concept_id == dependent_id,
                ConceptRelation.relation_type == "prerequisite",
            )
        ).scalars().all()

        for relation in relations:
            prerequisite_id = relation.from_concept_id
            if prerequisite_id in visited:
                continue
            concept = db.get(Concept, prerequisite_id)
            if not concept:
                continue
            if course_id is not None:
                has_course_evidence = db.execute(
                    select(Evidence.id)
                    .join(Chunk, Evidence.chunk_id == Chunk.id)
                    .join(Document, Chunk.document_id == Document.id)
                    .join(Source, Document.source_id == Source.id)
                    .where(Evidence.concept_id == prerequisite_id, Source.course_id == course_id)
                    .limit(1)
                ).first()
                if not has_course_evidence:
                    continue
            visited.add(prerequisite_id)
            next_depth = current_depth + 1
            results.append((concept, next_depth))
            queue.append((prerequisite_id, next_depth))

    return results


def prerequisite_evidence_chunks(
    db: Session,
    target_concept_names: list[str],
    depth: int,
    course_id: int | None = None,
    limit: int = 8,
) -> list[tuple[Chunk, float, str]]:
    targets = resolve_concepts(db, target_concept_names)
    chain = prerequisite_chain(db, [c.id for c in targets], depth=depth, course_id=course_id)
    if not chain:
        return []

    out: list[tuple[Chunk, float, str]] = []
    seen: set[int] = set()
    for concept, hop in chain:
        rows = db.execute(
            select(Evidence.chunk_id, Evidence.score)
            .where(Evidence.concept_id == concept.id)
            .order_by(Evidence.score.desc())
            .limit(4)
        ).all()
        for chunk_id, evidence_score in rows:
            if chunk_id in seen:
                continue
            chunk = db.get(Chunk, chunk_id)
            if not chunk or chunk.chunk_kind != "leaf":
                continue
            if course_id is not None:
                source = db.execute(
                    select(Source)
                    .join(Document, Source.id == Document.source_id)
                    .join(Chunk, Document.id == Chunk.document_id)
                    .where(Chunk.id == chunk.id)
                ).scalar_one_or_none()
                if not source or source.course_id != course_id:
                    continue
            seen.add(chunk.id)
            # Prerequisite evidence is intentionally lower-priority than direct retrieval.
            score = max(0.01, float(evidence_score or 1.0)) * (0.35 ** max(hop - 1, 0))
            out.append((chunk, score, concept.canonical_name))
            if len(out) >= limit:
                return out
    return out


def extract_group_relations(db: Session, ai: AIProvider, group_id: int) -> int:
    concepts = db.execute(
        select(Concept).where(Concept.group_id == group_id).order_by(Concept.canonical_name)
    ).scalars().all()
    if len(concepts) < 2 or not settings.concept_relationships:
        return 0

    concept_lines = []
    by_name = {}
    for concept in concepts[:40]:
        concept_lines.append(
            f"- {concept.canonical_name}: {concept.description or ''} | aliases={concept.aliases_json or '[]'}"
        )
        by_name[concept.canonical_name.strip().lower()] = concept

    prompt = f"""Identify useful learning relationships among these concepts.
Only create a prerequisite relation when the FIRST concept is needed to understand or use the SECOND concept.
Return no relationship merely because topics are related.
Group: {group_id}
Concepts:
{chr(10).join(concept_lines)}
Return JSON with a relationships array. Each relationship must use:
- from_concept: exact concept name from the list
- to_concept: exact concept name from the list
- type: prerequisite, related, or contradicts
- confidence: 0 to 1
"""

    schema = {
        "type": "object",
        "properties": {
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from_concept": {"type": "string"},
                        "to_concept": {"type": "string"},
                        "type": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["from_concept", "to_concept", "type", "confidence"],
                },
            }
        },
        "required": ["relationships"],
    }

    try:
        result = ai.generate_json(prompt, task="relationship_extraction", schema=schema)
    except Exception:
        return 0

    created = 0
    for item in result.get("relationships", []):
        from_name = str(item.get("from_concept", "")).strip().lower()
        to_name = str(item.get("to_concept", "")).strip().lower()
        rel_type = str(item.get("type", "")).strip().lower()
        if not from_name or not to_name or from_name == to_name or rel_type not in RELATION_TYPES:
            continue
        source = by_name.get(from_name)
        target = by_name.get(to_name)
        if not source or not target:
            continue
        confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        if confidence < settings.relationship_min_confidence:
            continue

        exists = db.execute(
            select(ConceptRelation).where(
                ConceptRelation.from_concept_id == source.id,
                ConceptRelation.to_concept_id == target.id,
                ConceptRelation.relation_type == rel_type,
            )
        ).scalar_one_or_none()
        if exists:
            exists.confidence = max(exists.confidence, confidence)
            continue
        db.add(
            ConceptRelation(
                from_concept_id=source.id,
                to_concept_id=target.id,
                relation_type=rel_type,
                confidence=confidence,
            )
        )
        created += 1
    return created
