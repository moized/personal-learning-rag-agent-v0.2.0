"""Learner state: events, concept mastery, misconceptions and SM-2 spaced review."""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, UTC

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import (
    Chunk, Concept, ConceptMastery, ConceptRelation, Document, Evidence,
    LearnerEvent, Misconception, Source,
)
from .spaces import in_space

MASTERED = 0.7
WEAK = 0.5
LEARNING_RATE = 0.35
CORRECT_THRESHOLD = 0.6
# At the scheduled review time, retention has decayed to exp(-1/STABILITY_FACTOR) (~72%).
STABILITY_FACTOR = 3.0


def now() -> datetime:
    """Naive UTC, matching how SQLite hands datetimes back."""
    return datetime.now(UTC).replace(tzinfo=None)


def sm2_update(ease: float, interval: float, repetitions: int, score: float) -> tuple[float, float, int]:
    """SM-2: map a 0-1 score to quality 0-5 and return (ease, interval_days, repetitions)."""
    quality = round(max(0.0, min(1.0, score)) * 5)
    ease = max(1.3, ease + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    if quality < 3:
        repetitions, interval = 0, 1.0
    else:
        repetitions += 1
        interval = 1.0 if repetitions == 1 else 6.0 if repetitions == 2 else interval * ease
    return ease, interval, repetitions


def effective_mastery(row: ConceptMastery | None, at: datetime | None = None) -> float:
    """Stored mastery decayed by the forgetting curve since it was last seen."""
    if row is None or row.last_seen_at is None:
        return 0.0
    elapsed = max(0.0, ((at or now()) - row.last_seen_at).total_seconds() / 86400)
    stability = max(row.interval_days, 1.0) * STABILITY_FACTOR
    return max(0.0, row.mastery * math.exp(-elapsed / stability))


def log_event(
    db: Session,
    event_type: str,
    *,
    concept_id: int | None = None,
    course_id: int | None = None,
    quiz_item_id: int | None = None,
    score: float | None = None,
    payload: dict | None = None,
) -> LearnerEvent:
    event = LearnerEvent(
        event_type=event_type,
        concept_id=concept_id,
        course_id=course_id,
        quiz_item_id=quiz_item_id,
        score=score,
        payload_json=json.dumps(payload, ensure_ascii=False) if payload else None,
        created_at=now(),
    )
    db.add(event)
    db.flush()
    return event


def record_result(
    db: Session,
    concept_id: int,
    score: float,
    *,
    event_type: str = "quiz_answer",
    course_id: int | None = None,
    quiz_item_id: int | None = None,
    payload: dict | None = None,
) -> ConceptMastery:
    """Log a scored attempt and update mastery and the review schedule for the concept."""
    if db.get(Concept, concept_id) is None:
        raise ValueError(f"Concept not found: {concept_id}")
    score = max(0.0, min(1.0, float(score)))
    log_event(db, event_type, concept_id=concept_id, course_id=course_id,
              quiz_item_id=quiz_item_id, score=score, payload=payload)
    at = now()
    row = db.execute(select(ConceptMastery).where(ConceptMastery.concept_id == concept_id)).scalar_one_or_none()
    if row is None:
        row = ConceptMastery(concept_id=concept_id, mastery=0.0, attempts=0, correct_count=0,
                             ease=2.5, interval_days=0.0, repetitions=0)
        db.add(row)
    prior = effective_mastery(row, at)
    row.mastery = prior + LEARNING_RATE * (score - prior)
    row.attempts += 1
    row.correct_count += int(score >= CORRECT_THRESHOLD)
    row.ease, row.interval_days, row.repetitions = sm2_update(row.ease, row.interval_days, row.repetitions, score)
    row.last_seen_at = at
    row.next_due_at = at + timedelta(days=row.interval_days)
    db.flush()
    return row


def add_misconception(db: Session, concept_id: int, description: str) -> Misconception:
    description = " ".join(description.split())
    key = description.casefold()
    existing = db.execute(
        select(Misconception).where(Misconception.concept_id == concept_id, Misconception.resolved.is_(False))
    ).scalars().all()
    for item in existing:
        if " ".join(item.description.split()).casefold() == key:
            item.occurrences += 1
            item.last_seen_at = now()
            db.flush()
            return item
    item = Misconception(concept_id=concept_id, description=description, first_seen_at=now(), last_seen_at=now())
    db.add(item)
    db.flush()
    return item


def open_misconceptions(db: Session, concept_ids: list[int]) -> list[Misconception]:
    if not concept_ids:
        return []
    return db.execute(
        select(Misconception)
        .where(Misconception.concept_id.in_(concept_ids), Misconception.resolved.is_(False))
        .order_by(Misconception.occurrences.desc(), Misconception.last_seen_at.desc())
    ).scalars().all()


def course_concept_ids(db: Session, course_id: int | None) -> list[int]:
    """Concepts with evidence in the course (or in any source when course_id is None)."""
    stmt = select(Evidence.concept_id).distinct()
    if course_id is not None:
        stmt = (
            stmt.join(Chunk, Evidence.chunk_id == Chunk.id)
            .join(Document, Chunk.document_id == Document.id)
            .join(Source, Document.source_id == Source.id)
            .where(in_space(course_id))
        )
    return [row[0] for row in db.execute(stmt).all()]


def _prerequisite_map(db: Session, concept_ids: list[int]) -> dict[int, list[int]]:
    """dependent concept -> its direct prerequisites, among the given concepts."""
    if not concept_ids:
        return {}
    ids = set(concept_ids)
    rows = db.execute(
        select(ConceptRelation.to_concept_id, ConceptRelation.from_concept_id).where(
            ConceptRelation.relation_type == "prerequisite",
            ConceptRelation.evidence_json.is_not(None),
            ConceptRelation.to_concept_id.in_(ids),
            ConceptRelation.from_concept_id.in_(ids),
        )
    ).all()
    out: dict[int, list[int]] = {}
    for dependent, prerequisite in rows:
        out.setdefault(dependent, []).append(prerequisite)
    return out


def _status(row: ConceptMastery | None, at: datetime) -> str:
    if row is None or row.attempts == 0:
        return "unseen"
    if row.next_due_at is not None and row.next_due_at <= at:
        return "due"
    return "mastered" if effective_mastery(row, at) >= MASTERED else "learning"


def mastery_overview(db: Session, course_id: int | None = None) -> list[dict]:
    ids = course_concept_ids(db, course_id)
    if not ids:
        return []
    at = now()
    concepts = {c.id: c for c in db.execute(select(Concept).where(Concept.id.in_(ids))).scalars().all()}
    rows = {r.concept_id: r for r in db.execute(
        select(ConceptMastery).where(ConceptMastery.concept_id.in_(ids))).scalars().all()}
    out = []
    for cid, concept in sorted(concepts.items(), key=lambda kv: kv[1].canonical_name.casefold()):
        row = rows.get(cid)
        attempts = row.attempts if row else 0
        out.append({
            "concept_id": cid,
            "concept": concept.canonical_name,
            "mastery": round(effective_mastery(row, at), 3),
            "confidence": round(attempts / (attempts + 2), 3),
            "attempts": attempts,
            "status": _status(row, at),
            "next_due_at": row.next_due_at.isoformat() if row and row.next_due_at else None,
        })
    return out


def due_reviews(db: Session, course_id: int | None = None, limit: int = 20) -> list[dict]:
    ids = set(course_concept_ids(db, course_id))
    if not ids:
        return []
    at = now()
    rows = db.execute(
        select(ConceptMastery)
        .where(ConceptMastery.concept_id.in_(ids), ConceptMastery.next_due_at <= at)
        .order_by(ConceptMastery.next_due_at)
        .limit(limit)
    ).scalars().all()
    out = []
    for row in rows:
        concept = db.get(Concept, row.concept_id)
        out.append({
            "concept_id": row.concept_id,
            "concept": concept.canonical_name if concept else str(row.concept_id),
            "mastery": round(effective_mastery(row, at), 3),
            "due_at": row.next_due_at.isoformat(),
            "overdue_days": round((at - row.next_due_at).total_seconds() / 86400, 2),
        })
    return out


def recommend_next(db: Session, course_id: int | None = None, limit: int = 5) -> list[dict]:
    """Rank what to study next: overdue reviews, then weak concepts, then newly unlocked ones.

    A concept is "ready" only when every prerequisite is mastered, so the learner is never
    pointed at material whose foundations are still missing. Blocked concepts are skipped;
    a due or weak concept with weak prerequisites reports them in `blocking_prerequisites`.
    """
    ids = course_concept_ids(db, course_id)
    if not ids:
        return []
    at = now()
    concepts = {c.id: c for c in db.execute(select(Concept).where(Concept.id.in_(ids))).scalars().all()}
    rows = {r.concept_id: r for r in db.execute(
        select(ConceptMastery).where(ConceptMastery.concept_id.in_(ids))).scalars().all()}
    prerequisites = _prerequisite_map(db, ids)
    dependents: dict[int, int] = {}
    for prereq_list in prerequisites.values():
        for pid in prereq_list:
            dependents[pid] = dependents.get(pid, 0) + 1

    suggestions: list[tuple[float, dict]] = []
    for cid, concept in concepts.items():
        row = rows.get(cid)
        mastery = effective_mastery(row, at)
        weak_prereqs = [
            concepts[p].canonical_name for p in prerequisites.get(cid, [])
            if p in concepts and effective_mastery(rows.get(p), at) < MASTERED
        ]
        status = _status(row, at)
        if status == "due":
            overdue = (at - row.next_due_at).total_seconds() / 86400
            priority, reason = 3.0 + min(overdue, 10) * 0.1, "review due"
        elif status == "learning" and mastery < WEAK:
            priority, reason = 2.0 + (WEAK - mastery), "weak - needs practice"
        elif status == "unseen" and not weak_prereqs:
            priority, reason = 1.0 + min(dependents.get(cid, 0), 5) * 0.05, "ready to learn"
        else:
            continue
        suggestions.append((priority, {
            "concept_id": cid,
            "concept": concept.canonical_name,
            "reason": reason,
            "mastery": round(mastery, 3),
            "blocking_prerequisites": weak_prereqs,
        }))
    suggestions.sort(key=lambda item: (-item[0], item[1]["concept"].casefold()))
    return [item for _, item in suggestions[:limit]]
