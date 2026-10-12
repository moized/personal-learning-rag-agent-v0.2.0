"""Knowledge graph primitives for dependency-aware retrieval and learning navigation."""
from __future__ import annotations

import json
import math
import re
from collections import deque

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import Chunk, ChunkRelation, Concept, ConceptRelation, Evidence, Document, Source
from ..providers.base import AIProvider
from ..config import settings
from .reliability import is_retryable_error
from .spaces import in_space


RELATION_TYPES = {"prerequisite", "parallel", "related", "contradicts"}


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
    normalized = {re.sub(r"\s+", " ", str(n).strip().lower()) for n in names if str(n).strip()}
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
        candidate_norm = {re.sub(r"\s+", " ", c.strip().lower()) for c in candidates if c and c.strip()}
        if candidate_norm & normalized:
            matched.append(concept)
            continue
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
                ConceptRelation.evidence_json.is_not(None),
            )
        ).scalars().all()

        for relation in relations:
            if course_id is not None:
                try:
                    relation_evidence = json.loads(relation.evidence_json or "{}")
                    evidence_ids = [int(chunk_id) for chunk_id in relation_evidence.get("chunk_ids", [])]
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if not evidence_ids:
                    continue
                supported_in_course = db.execute(
                    select(Chunk.id)
                    .join(Document, Chunk.document_id == Document.id)
                    .join(Source, Document.source_id == Source.id)
                    .where(Chunk.id.in_(evidence_ids), in_space(course_id))
                    .limit(1)
                ).first()
                if not supported_in_course:
                    continue
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
                    .where(Evidence.concept_id == prerequisite_id, in_space(course_id))
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
            .join(Chunk, Evidence.chunk_id == Chunk.id)
            .join(Document, Chunk.document_id == Document.id)
            .join(Source, Document.source_id == Source.id)
            .where(Evidence.concept_id == concept.id, Chunk.chunk_kind == "leaf", Chunk.embedding_status == "done", Source.status == "done")
            .order_by(Evidence.score.desc())
            .limit(4)
        ).all()
        for chunk_id, evidence_score in rows:
            if chunk_id in seen:
                continue
            chunk = db.get(Chunk, chunk_id)
            if not chunk or chunk.chunk_kind != "leaf" or chunk.embedding_status != "done":
                continue
            if course_id is not None:
                source = db.execute(
                    select(Source)
                    .join(Document, Source.id == Document.source_id)
                    .join(Chunk, Document.id == Chunk.document_id)
                    .where(Chunk.id == chunk.id, in_space(course_id), Source.status == "done")
                ).scalar_one_or_none()
                if not source:
                    continue
            seen.add(chunk.id)
            # Prerequisite evidence is intentionally lower-priority than direct retrieval.
            score = max(0.01, float(evidence_score or 1.0)) * (0.35 ** max(hop - 1, 0))
            out.append((chunk, score, concept.canonical_name))
            if len(out) >= limit:
                return out
    return out


def _would_create_prerequisite_cycle(
    db: Session,
    prerequisite_id: int,
    dependent_id: int,
) -> bool:
    """Reject an edge when the dependent already reaches the proposed prerequisite."""
    queue = deque([dependent_id])
    visited = {dependent_id}
    while queue:
        current = queue.popleft()
        if current == prerequisite_id:
            return True
        relations = db.execute(
            select(ConceptRelation).where(
                ConceptRelation.from_concept_id == current,
                ConceptRelation.relation_type == "prerequisite",
                ConceptRelation.evidence_json.is_not(None),
            )
        ).scalars().all()
        for relation in relations:
            if relation.to_concept_id not in visited:
                visited.add(relation.to_concept_id)
                queue.append(relation.to_concept_id)
    return False


def extract_concept_relations(
    db: Session,
    ai: AIProvider,
    concept_ids: set[int],
    scope_label: str,
) -> int:
    concepts = db.execute(
        select(Concept).where(Concept.id.in_(concept_ids)).order_by(Concept.canonical_name)
    ).scalars().all()
    if len(concepts) < 2 or not settings.concept_relationships:
        return 0

    concept_lines = []
    by_name = {}
    evidence_chunks: dict[int, Chunk] = {}
    for concept in concepts[:40]:
        evidence_rows = db.execute(
            select(Evidence, Chunk)
            .join(Chunk, Evidence.chunk_id == Chunk.id)
            .where(Evidence.concept_id == concept.id, Chunk.chunk_kind == "leaf")
            .order_by(Evidence.score.desc(), Chunk.id)
            .limit(2)
        ).all()
        excerpts = []
        for _evidence, chunk in evidence_rows:
            evidence_chunks[chunk.id] = chunk
            excerpt = re.sub(r"\s+", " ", chunk.text).strip()[:700]
            excerpts.append(f"[chunk:{chunk.id}] {excerpt}")
        concept_lines.append(
            f"- {concept.canonical_name}: {concept.description or ''} | aliases={concept.aliases_json or '[]'}"
            + (f"\n  Source evidence: {' '.join(excerpts)}" if excerpts else "\n  Source evidence: none")
        )
        by_name[concept.canonical_name.strip().lower()] = concept

    prompt = f"""Extract source-supported learning relationships among these concepts.
Use only the source evidence shown below. Concept descriptions are labels, not proof of a dependency.
Create a prerequisite edge from the prerequisite to the dependent concept only when the evidence shows that the first concept is required to understand or use the second.
Do not infer prerequisites from presentation order, a common curriculum, or a possible processing pipeline.
Optional preprocessing is not a prerequisite: for example, stemming may be used before vectorization, but vectorization or embeddings do not require stemming.
Different representation methods that solve a similar task (such as sparse frequency vectors and dense semantic embeddings) are parallel approaches, not prerequisites of one another.
Use "related" for a meaningful connection without a dependency or parallel-method relationship. Use "contradicts" only for a source-supported disagreement.
Every relationship must cite one or more evidence chunk IDs that directly support that relationship. If the excerpts do not establish a relationship, omit it.
Concept scope: {scope_label}
Concepts:
{chr(10).join(concept_lines)}
Return JSON with a relationships array. Each relationship must use:
- from_concept: exact concept name from the list
- to_concept: exact concept name from the list
- type: prerequisite, parallel, related, or contradicts
- confidence: 0 to 1
- evidence_chunk_ids: one or more chunk IDs shown in the evidence
- rationale: a short explanation grounded in those chunks
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
                        "evidence_chunk_ids": {"type": "array", "items": {"type": "integer"}},
                        "rationale": {"type": "string"},
                    },
                    "required": ["from_concept", "to_concept", "type", "confidence", "evidence_chunk_ids", "rationale"],
                },
            }
        },
        "required": ["relationships"],
    }

    try:
        result = ai.generate_json(prompt, task="relationship_extraction", schema=schema)
    except Exception as exc:
        if is_retryable_error(exc):
            raise
        return 0

    concept_ids = {concept.id for concept in by_name.values()}
    # Legacy graph edges had no source provenance. Remove them before cycle checks
    # so they cannot block a corrected, evidence-backed edge.
    legacy_edges = db.execute(
        select(ConceptRelation).where(
            ConceptRelation.evidence_json.is_(None),
            ConceptRelation.from_concept_id.in_(concept_ids),
            ConceptRelation.to_concept_id.in_(concept_ids),
        )
    ).scalars().all()
    for edge in legacy_edges:
        db.delete(edge)
    db.flush()

    relationships = result.get("relationships", [])
    if not isinstance(relationships, list):
        relationships = []
    created = 0
    for item in relationships:
        if not isinstance(item, dict):
            continue
        from_name = str(item.get("from_concept", "")).strip().lower()
        to_name = str(item.get("to_concept", "")).strip().lower()
        rel_type = str(item.get("type", "")).strip().lower()
        if not from_name or not to_name or from_name == to_name or rel_type not in RELATION_TYPES:
            continue
        source = by_name.get(from_name)
        target = by_name.get(to_name)
        if not source or not target:
            continue
        try:
            evidence_ids = list(dict.fromkeys(int(chunk_id) for chunk_id in item.get("evidence_chunk_ids", [])))
        except (TypeError, ValueError):
            continue
        evidence_ids = [chunk_id for chunk_id in evidence_ids if chunk_id in evidence_chunks]
        if not evidence_ids:
            continue
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            continue
        if confidence < settings.relationship_min_confidence:
            continue

        if rel_type == "prerequisite" and _would_create_prerequisite_cycle(
            db, prerequisite_id=source.id, dependent_id=target.id
        ):
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
            try:
                old_evidence = json.loads(exists.evidence_json or "{}")
            except json.JSONDecodeError:
                old_evidence = {}
            old_ids = old_evidence.get("chunk_ids", []) if isinstance(old_evidence, dict) else []
            if not isinstance(old_ids, list):
                old_ids = []
            exists.evidence_json = json.dumps({
                "chunk_ids": list(dict.fromkeys([*old_ids, *evidence_ids]))[:8],
                "rationale": str(item.get("rationale", ""))[:1000],
                "method": "source-grounded-v1",
            }, ensure_ascii=False)
            continue
        db.add(
            ConceptRelation(
                from_concept_id=source.id,
                to_concept_id=target.id,
                relation_type=rel_type,
                confidence=confidence,
                evidence_json=json.dumps({
                    "chunk_ids": evidence_ids[:8],
                    "rationale": str(item.get("rationale", ""))[:1000],
                    "method": "source-grounded-v1",
                }, ensure_ascii=False),
            )
        )
        # Flush so a repeated pair later in this reply merges and the cycle check sees this edge.
        db.flush()
        created += 1
    return created


def extract_group_relations(db: Session, ai: AIProvider, group_id: int) -> int:
    """Compatibility wrapper for refreshing one semantic group."""
    concept_ids = set(db.execute(
        select(Concept.id).where(Concept.group_id == group_id)
    ).scalars().all())
    return extract_concept_relations(db, ai, concept_ids, f"semantic group {group_id}")


def extract_source_relations(db: Session, ai: AIProvider, source_id: int, before_window=None) -> int:
    """Extract relations among one source's concepts, in batches of concepts that first appear close together.

    One call sees at most 40 concepts, so a large source is processed in reading order
    rather than truncated to the first 40 names alphabetically.
    """
    first_seen = func.min(Document.id * 1_000_000 + Chunk.chunk_index)
    ordered = db.execute(
        select(Evidence.concept_id)
        .join(Chunk, Evidence.chunk_id == Chunk.id)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.source_id == source_id, Chunk.chunk_kind == "leaf")
        .group_by(Evidence.concept_id)
        .order_by(first_seen, Evidence.concept_id)
    ).scalars().all()
    size = max(2, min(40, settings.relationship_batch_size))
    overlap = size // 4
    if len(ordered) <= size:
        windows = [ordered]
    else:
        # Even, overlapping windows: no batch is left with a single concept, and concepts
        # near a window boundary are also compared with their neighbours in the next window.
        count = math.ceil((len(ordered) - overlap) / (size - overlap))
        step = math.ceil((len(ordered) - overlap) / count)
        windows = [ordered[i * step:i * step + step + overlap] for i in range(count)]
    created = 0
    if len(windows) > 1 and len(windows[-1]) < 2:
        windows[-1] = ordered[-size:]  # tiny batch sizes: never leave the last concept alone
    for window in windows:
        if len(window) >= 2:
            if before_window is not None:
                before_window()  # e.g. a background job's heartbeat before each AI call
            created += extract_concept_relations(db, ai, set(window), f"source {source_id}")
            # Commit per window: no write lock is held during the next AI call, and windows
            # that finished survive a failure in a later one.
            db.commit()
    return created
