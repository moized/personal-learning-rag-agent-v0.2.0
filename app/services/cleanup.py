"""Removing material and study spaces completely.

Removing a source deletes it and everything LearnOS made from it: passages, embeddings, sections,
key ideas only it supported (with their practice questions and mistakes), prepared study sessions
that may quote it, topic groups left empty, and logs that used its passages. Deleted text is also
erased from the database file: SQLite overwrites deleted content (secure_delete), and the worker
compacts the file afterwards so nothing readable is left in its free space or its write-ahead log.
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import (
    AgentTrace, Chunk, ChunkRelation, Concept, ConceptMastery, ConceptRelation, Course, Document, Evidence, Group, Job,
    LearnerEvent, MaintenanceJob, Misconception, QuizItem, Section, Source, SourceRelationship, SourceSpace, StudyPlan,
    StudyPlanNode, StudyProgress, TopicDetail,
)
from .spaces import spaces_of

COMPACT_TASK = "compact_storage"


def delete_sources(db: Session, sources: list[Source]) -> tuple[list[Path], set[int]]:
    """Delete these sources and everything made from them. Returns (files to delete, spaces they were in)."""
    source_ids = {source.id for source in sources}
    spaces = spaces_of(db, source_ids)
    document_ids = [row[0] for row in db.execute(select(Document.id).where(Document.source_id.in_(source_ids))).all()]
    chunk_rows = db.execute(select(Chunk.id, Chunk.parent_chunk_id).where(Chunk.document_id.in_(document_ids))).all() if document_ids else []
    chunk_ids = {row.id for row in chunk_rows}
    db.execute(delete(Section).where(Section.source_id.in_(source_ids)))
    if chunk_ids:
        db.execute(delete(ChunkRelation).where(or_(ChunkRelation.from_chunk_id.in_(chunk_ids), ChunkRelation.to_chunk_id.in_(chunk_ids))))
        db.execute(delete(Evidence).where(Evidence.chunk_id.in_(chunk_ids)))
        # Remove leaf chunks before their parent windows to respect the self-reference.
        remaining = {row.id: row.parent_chunk_id for row in chunk_rows}
        while remaining:
            parents = {parent for parent in remaining.values() if parent is not None}
            removable = set(remaining).difference(parents) or set(remaining)
            db.execute(delete(Chunk).where(Chunk.id.in_(removable)))
            for chunk_id in removable:
                remaining.pop(chunk_id, None)
        orphans = [row[0] for row in db.execute(
            select(Concept.id).where(~select(Evidence.id).where(Evidence.concept_id == Concept.id).exists())).all()]
        if orphans:
            # Practice made from key ideas nothing supports any more: questions, answers, mistakes, mastery.
            quiz_ids = [row[0] for row in db.execute(select(QuizItem.id).where(QuizItem.concept_id.in_(orphans))).all()]
            db.execute(delete(LearnerEvent).where(or_(LearnerEvent.concept_id.in_(orphans),
                                                      LearnerEvent.quiz_item_id.in_(quiz_ids) if quiz_ids else False)))
            db.execute(delete(QuizItem).where(QuizItem.concept_id.in_(orphans)))
            db.execute(delete(Misconception).where(Misconception.concept_id.in_(orphans)))
            db.execute(delete(ConceptMastery).where(ConceptMastery.concept_id.in_(orphans)))
            db.execute(delete(ConceptRelation).where(or_(ConceptRelation.from_concept_id.in_(orphans), ConceptRelation.to_concept_id.in_(orphans))))
            db.execute(delete(Concept).where(Concept.id.in_(orphans)))
        # Logs of questions answered from its passages.
        for trace in db.execute(select(AgentTrace).where(AgentTrace.result_json.is_not(None))).scalars().all():
            try:
                used = set(json.loads(trace.result_json or "{}").get("retrieved_chunk_ids") or [])
            except (TypeError, ValueError, AttributeError):
                continue
            if used & chunk_ids:
                db.delete(trace)
    if document_ids:
        db.execute(delete(Document).where(Document.id.in_(document_ids)))
    db.execute(delete(Job).where(Job.source_id.in_(source_ids)))
    db.execute(delete(SourceRelationship).where(or_(SourceRelationship.source_a_id.in_(source_ids), SourceRelationship.source_b_id.in_(source_ids))))
    # Prepared study sessions quote passages of the spaces they were made in (and of All sources).
    db.execute(delete(TopicDetail).where(TopicDetail.scope_key.in_({f"course:{c}" for c in spaces} | {"all"})))
    db.execute(delete(Group).where(~select(Chunk.id).where(Chunk.group_id == Group.id).exists()))
    files: list[Path] = []
    upload_root = settings.upload_path.resolve()
    for source in sources:
        if source.local_path:
            candidate = Path(source.local_path).resolve()
            if candidate.is_relative_to(upload_root):
                files.append(candidate)
    db.execute(delete(SourceSpace).where(SourceSpace.source_id.in_(source_ids)))
    db.execute(delete(Source).where(Source.id.in_(source_ids)))
    db.flush()
    return files, spaces


def forget_scope(db: Session, scope: str) -> None:
    """A study plan's scope has no material any more: its plan, prepared sessions and progress go too."""
    plan = db.execute(select(StudyPlan).where(StudyPlan.scope_key == scope)).scalar_one_or_none()
    if plan is not None:
        db.execute(delete(StudyPlanNode).where(StudyPlanNode.plan_id == plan.id))
        db.delete(plan)
    db.execute(delete(TopicDetail).where(TopicDetail.scope_key == scope))
    db.execute(delete(StudyProgress).where(StudyProgress.scope_key == scope))


def forget_empty_scopes(db: Session) -> None:
    """Prepared sessions, progress and plans left for spaces (or All sources) that have no material any more."""
    scopes = set(db.execute(select(TopicDetail.scope_key)).scalars().all()) | set(db.execute(select(StudyProgress.scope_key)).scalars().all()) \
        | set(db.execute(select(StudyPlan.scope_key)).scalars().all())
    for scope in scopes:
        if scope == "all":
            empty = db.execute(select(Source.id).limit(1)).first() is None
        elif scope.startswith("course:") and scope[7:].isdigit():
            course_id = int(scope[7:])
            empty = db.get(Course, course_id) is None or (
                db.execute(select(Source.id).where(Source.course_id == course_id).limit(1)).first() is None
                and db.execute(select(SourceSpace.id).where(SourceSpace.course_id == course_id).limit(1)).first() is None)
        else:
            continue
        if empty:
            forget_scope(db, scope)


def remove_space(db: Session, course: Course, *, with_material: bool) -> list[Path]:
    """Delete a study space. Its material is deleted too, or kept in All sources. Returns files to delete."""
    files: list[Path] = []
    filed = db.execute(select(Source).where(Source.course_id == course.id)).scalars().all()
    if with_material and filed:
        files, _ = delete_sources(db, filed)
    else:
        for source in filed:
            source.course_id = None
            source.assignment_mode, source.assignment_status = "explicit_unassigned", "unassigned"
            source.assignment_reason = f'Its study space "{course.name}" was deleted; it stays in All sources.'
    db.execute(delete(SourceSpace).where(SourceSpace.course_id == course.id))
    for source in db.execute(select(Source).where(Source.placement_json.is_not(None))).scalars().all():
        try:
            choices = json.loads(source.placement_json or "{}")
        except (TypeError, ValueError):
            continue
        if isinstance(choices, dict) and (course.id in (choices.get("kept_out_of") or []) or choices.get("kept_in") == course.id):
            choices["kept_out_of"] = [c for c in choices.get("kept_out_of") or [] if c != course.id]
            if choices.get("kept_in") == course.id:
                choices.pop("kept_in")
            source.placement_json = json.dumps(choices)
    for model in (LearnerEvent, QuizItem):
        for row in db.execute(select(model).where(model.course_id == course.id)).scalars().all():
            row.course_id = None
    forget_scope(db, f"course:{course.id}")
    db.delete(course)
    db.flush()
    return files


def remove_empty_spaces(db: Session, course_ids) -> list[int]:
    """Spaces LearnOS created by itself are removed once nothing is left in them; the learner's own stay."""
    removed = []
    for course_id in sorted({c for c in course_ids if c is not None}):
        course = db.get(Course, course_id)
        if course is None or course.created_by != "learnos":
            continue
        if db.execute(select(Source.id).where(Source.course_id == course_id).limit(1)).first() is None and \
                db.execute(select(SourceSpace.id).where(SourceSpace.course_id == course_id).limit(1)).first() is None:
            remove_space(db, course, with_material=False)
            removed.append(course_id)
    return removed


def request_compaction(db: Session) -> None:
    """Compact the database file in the background, so deleted text is not left in it."""
    queued = db.execute(select(MaintenanceJob.id).where(MaintenanceJob.task == COMPACT_TASK, MaintenanceJob.status == "queued")).first()
    if queued is None:
        db.add(MaintenanceJob(task=COMPACT_TASK, status="queued", progress_total=1, items_json=json.dumps([{"kind": "compact", "id": 0}])))
        db.flush()


def delete_files(files: list[Path]) -> None:
    for path in files:
        if path.exists():
            path.unlink()
