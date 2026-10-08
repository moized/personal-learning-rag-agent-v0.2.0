"""Learner-intelligence endpoints: quizzes, mastery, spaced review, misconceptions."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from ..db import session_scope
from ..models import Concept, Misconception, QuizItem
from ..providers.factory import provider
from ..services.knowledge_graph import resolve_concepts
from ..services.learner import (
    due_reviews, mastery_overview, recommend_next, record_result,
)
from ..services.quiz import generate_quiz_items, grade_answer

router = APIRouter(tags=["learner"])


class QuizGenerateRequest(BaseModel):
    concept_id: int | None = None
    concept_name: str | None = Field(default=None, max_length=255)
    course_id: int | None = None
    count: int = Field(default=1, ge=1, le=5)


class QuizAnswerRequest(BaseModel):
    answer: str = Field(max_length=6000)


class SelfRatingRequest(BaseModel):
    concept_id: int
    rating: int = Field(ge=0, le=5, description="SM-2 style recall quality, 0 (blackout) to 5 (perfect)")
    course_id: int | None = None


@router.post("/quiz/generate")
def quiz_generate(req: QuizGenerateRequest):
    with session_scope() as db:
        concept = db.get(Concept, req.concept_id) if req.concept_id is not None else None
        if concept is None and req.concept_name:
            matches = resolve_concepts(db, [req.concept_name])
            concept = matches[0] if matches else None
        if concept is None:
            raise HTTPException(404, "Concept not found")
        try:
            items = generate_quiz_items(db, provider(), concept, req.count, req.course_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        # The reference answer is withheld until the learner submits an answer.
        return {
            "concept_id": concept.id,
            "concept": concept.canonical_name,
            "items": [{"id": i.id, "question": i.question} for i in items],
        }


@router.post("/quiz/{item_id}/answer")
def quiz_answer(item_id: int, req: QuizAnswerRequest):
    with session_scope() as db:
        item = db.get(QuizItem, item_id)
        if item is None:
            raise HTTPException(404, "Quiz item not found")
        return grade_answer(db, provider(), item, req.answer)


@router.post("/learner/self-rating")
def self_rating(req: SelfRatingRequest):
    with session_scope() as db:
        try:
            row = record_result(db, req.concept_id, req.rating / 5, event_type="self_rating",
                                course_id=req.course_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc))
        return {"concept_id": row.concept_id, "mastery": round(row.mastery, 3),
                "next_due_at": row.next_due_at.isoformat() if row.next_due_at else None}


@router.get("/learner/mastery")
def mastery(course_id: int | None = None):
    with session_scope() as db:
        return {"concepts": mastery_overview(db, course_id)}


@router.get("/learner/due")
def due(course_id: int | None = None, limit: int = 20):
    with session_scope() as db:
        return {"due": due_reviews(db, course_id, max(1, min(limit, 100)))}


@router.get("/learner/next")
def next_up(course_id: int | None = None, limit: int = 5):
    with session_scope() as db:
        return {"next": recommend_next(db, course_id, max(1, min(limit, 20)))}


@router.get("/learner/misconceptions")
def misconceptions(concept_id: int | None = None, include_resolved: bool = False):
    with session_scope() as db:
        stmt = select(Misconception).order_by(Misconception.last_seen_at.desc())
        if concept_id is not None:
            stmt = stmt.where(Misconception.concept_id == concept_id)
        if not include_resolved:
            stmt = stmt.where(Misconception.resolved.is_(False))
        rows = db.execute(stmt).scalars().all()
        names = {c.id: c.canonical_name for c in db.execute(
            select(Concept).where(Concept.id.in_({r.concept_id for r in rows}))).scalars().all()} if rows else {}
        return {"misconceptions": [
            {"id": r.id, "concept_id": r.concept_id, "concept": names.get(r.concept_id),
             "description": r.description, "occurrences": r.occurrences, "resolved": r.resolved,
             "last_seen_at": r.last_seen_at.isoformat()}
            for r in rows
        ]}


@router.post("/learner/misconceptions/{misconception_id}/resolve")
def resolve_misconception(misconception_id: int):
    with session_scope() as db:
        row = db.get(Misconception, misconception_id)
        if row is None:
            raise HTTPException(404, "Misconception not found")
        row.resolved = True
        return {"id": row.id, "resolved": True}
