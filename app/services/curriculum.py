import json
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..models import Group, Concept, Evidence, Chunk, Document, Source, Course
from ..providers.base import AIProvider
from ..config import settings
from .retrieval import retrieve, group_candidates


def _course_context(db: Session, course_id: int | None) -> tuple[Course | None, list[tuple[str, list[str]]]]:
    course = db.get(Course, course_id) if course_id else None
    stmt = select(Chunk).join(Document, Chunk.document_id == Document.id).join(Source, Document.source_id == Source.id).where(Chunk.embedding_status == "done")
    if course_id:
        stmt = stmt.where(Source.course_id == course_id)
    chunk_ids = {c.id for c in db.execute(stmt).scalars().all()}
    rows = db.execute(select(Group, Concept, Evidence).join(Concept, Concept.group_id == Group.id).join(Evidence, Evidence.concept_id == Concept.id)).all()
    grouped: dict[str, set[str]] = {}
    for group, concept, evidence in rows:
        if evidence.chunk_id in chunk_ids:
            grouped.setdefault(group.name, set()).add(concept.canonical_name)
    return course, [(name, sorted(values)) for name, values in sorted(grouped.items())]


def build_syllabus(db: Session, ai: AIProvider, course_id: int | None = None) -> str:
    course, grouped = _course_context(db, course_id)
    if not grouped:
        return "No organized learning material is available for this scope yet."
    lines = [f"## {name}\n" + ", ".join(concepts[:80]) for name, concepts in grouped]
    outline = course.outline if course else None
    course_desc = course.description if course else None
    prompt = f"""Design a learning syllabus from the organized knowledge below.
Merge duplicate concepts across sources, preserve important distinctions, and order prerequisites before dependent topics.
Do not invent topics absent from the evidence. Treat the user's outline as a preference, not as evidence that a topic exists.
For every syllabus topic provide: id, title, purpose, prerequisites, and source_groups.
Course description: {course_desc or 'none'}
User outline: {outline or 'none'}
Organized concepts:
{chr(10).join(lines)}
Return a clean numbered syllabus. A compact table-like structure is preferred."""
    return ai.generate_text(prompt, task="syllabus")


def build_study_pack_prompt(db: Session, ai: AIProvider, course_id: int | None, topic: str, top_k: int = 12) -> tuple[str, list[dict]]:
    q = ai.embed([topic])[0]
    candidate_groups = group_candidates(db, q, None, limit=settings.retrieval_group_limit, course_id=course_id)
    scored = retrieve(db, ai, topic, candidate_groups, top_k=top_k, course_id=course_id)
    if len(scored) < max(2, top_k // 2):
        fallback = retrieve(db, ai, topic, None, top_k=top_k, course_id=course_id)
        seen = {c.id for c, _ in scored}
        scored.extend((c, s) for c, s in fallback if c.id not in seen)
        scored = scored[:top_k]
    if not scored:
        return "No relevant material was found for this topic in the selected course.", []

    evidence: list[str] = []
    refs: list[dict] = []
    for c, score in scored:
        doc = db.get(Document, c.document_id)
        src = db.get(Source, doc.source_id) if doc else None
        label = f"{src.name if src else 'source'} / {doc.title if doc else 'document'} / {doc.section or 'section'}"
        evidence.append(f"[{label} | chunk:{c.id}]\n{c.text}")
        refs.append({"chunk_id": c.id, "score": score, "source": src.name if src else None, "section": doc.section if doc else None})

    course = db.get(Course, course_id) if course_id else None
    prompt = f"""Prepare a rigorous study pack for a student who may paste it into another AI tutor.
Topic: {topic}
Course: {course.name if course else 'all material'}
Use only the supplied evidence. Synthesize repeated explanations instead of repeating them. Preserve disagreements or uncertainty.
Include source labels and [chunk:ID] citations next to factual claims where appropriate.
Structure:
1. Learning objective
2. Prerequisites
3. Core explanation
4. Key concepts and terminology
5. Connections and mental model
6. Worked/intuitive example only when supported
7. Common confusions and failure modes
8. Five self-check questions
9. Suggested next topic

Evidence:
{chr(10).join(evidence)}"""
    return ai.generate_text(prompt, task="study"), refs
