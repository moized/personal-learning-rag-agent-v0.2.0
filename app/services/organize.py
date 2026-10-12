"""Keeping the library organized without the learner having to ask.

The work older or interrupted sources still need: grouping passages, refreshing a source's profile,
filing sources LearnOS was asked to file, reading every passage for key ideas and sections, and
checking where each source belongs. The worker queues it by itself in the background (its own task,
so a run paused by the AI quota is not shown as something for the learner to fix); the Library's
"Check everything now" asks for it at once. A background run never repeats work it already tried
while nothing about that work has changed, so something that keeps failing costs nothing more.
"""
from __future__ import annotations

import hashlib
import json

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Chunk, Document, MaintenanceJob, Source, SourceRelationship
from .concepts import CONCEPT_PASS
from .reliability import profile_needs_refresh

ORGANIZE_TASK = "organize_existing"        # asked for in the Library
BACKGROUND_TASK = "organize_background"    # queued by the worker by itself
ORGANIZE_TASKS = (ORGANIZE_TASK, BACKGROUND_TASK)


def index_pending():
    """Leaves that still need reading for concepts (when enabled) or a section."""
    condition = Chunk.section_id.is_(None)
    if settings.concept_extraction:
        condition = condition | (Chunk.concept_pass < CONCEPT_PASS)
    return condition


def unread_counts(db: Session, source_id: int) -> tuple[int, int]:
    """(leaves not read for key ideas, leaves without a section) of a source."""
    base = select(func.count(Chunk.id)).join(Document, Chunk.document_id == Document.id).where(
        Document.source_id == source_id, Chunk.chunk_kind == "leaf")
    unread = db.execute(base.where(Chunk.concept_pass < CONCEPT_PASS)).scalar() if settings.concept_extraction else 0
    return int(unread or 0), int(db.execute(base.where(Chunk.section_id.is_(None))).scalar() or 0)


def _fingerprint(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def organization_items(db: Session, *, include_unassigned: bool, automatic: bool) -> list[dict]:
    chunk_ids = db.execute(select(Chunk.id).join(Document, Chunk.document_id == Document.id).join(Source, Document.source_id == Source.id)
                           .where(Chunk.chunk_kind == "leaf", Chunk.embedding_status == "done",
                                  (Chunk.group_id.is_(None)) | (Chunk.classification_status != "done"), Source.status == "done")
                           .order_by(Chunk.id)).scalars().all()
    done_sources = db.execute(select(Source).where(Source.status == "done").order_by(Source.id)).scalars().all()
    items: list[dict] = [{"kind": "chunk", "id": chunk_id, "fp": "1"} for chunk_id in chunk_ids]
    for source in done_sources:
        try:
            profile = json.loads(source.profile_json or "{}")
        except json.JSONDecodeError:
            profile = {}
        # Filing by metadata only for sources LearnOS was asked to file, or every unfiled one when that is asked
        # for explicitly (the background run and the Library's button never ask: a source kept unfiled stays so).
        should_assign = source.course_id is None and (source.assignment_mode == "auto" or include_unassigned)
        if profile_needs_refresh(profile) or should_assign:
            item = {"kind": "source", "id": source.id, "fp": _fingerprint(source.profile_json, source.course_id, source.assignment_mode)}
            if should_assign:
                item["auto_assign_unassigned"] = source.assignment_mode != "auto"
            items.append(item)
    # Sources whose passages were never read in full for key ideas, or that have no sections yet.
    unread = set(db.execute(
        select(Document.source_id).join(Chunk, Chunk.document_id == Document.id).join(Source, Document.source_id == Source.id)
        .where(Source.status == "done", Chunk.chunk_kind == "leaf", index_pending()).distinct()
    ).scalars().all())
    items += [{"kind": "concepts", "id": source_id, "fp": _fingerprint(*unread_counts(db, source_id))} for source_id in sorted(unread)]
    # Last, once profiles and key ideas are current: where each source belongs (pairs judged before cost nothing).
    # A background run only checks sources never compared with any other, and only when there is a filed source to compare with.
    filed = [source.id for source in done_sources if source.course_id is not None]
    if len(done_sources) > 1:
        compared: set[int] = set()
        if automatic:
            for a, b in db.execute(select(SourceRelationship.source_a_id, SourceRelationship.source_b_id)).all():
                compared.update((a, b))
        for source in done_sources:
            if source.id in compared or (automatic and not [other for other in filed if other != source.id]):
                continue
            items.append({"kind": "placement", "id": source.id, "fp": _fingerprint(len(filed))})
    return items


def _already_tried(db: Session) -> set[tuple]:
    """Items the last background run finished without an error, with what they were tried on."""
    last = db.execute(select(MaintenanceJob).where(MaintenanceJob.task == BACKGROUND_TASK).order_by(MaintenanceJob.id.desc()).limit(1)).scalar_one_or_none()
    if last is None:
        return set()
    try:
        items = json.loads(last.items_json or "[]")
        errors = json.loads(last.errors_json or "{}")
    except json.JSONDecodeError:
        return set()
    return {(item.get("kind"), item.get("id"), item.get("fp")) for item in items[:last.progress_current]
            if isinstance(item, dict) and f"{item.get('kind')}:{item.get('id')}" not in errors}


def queue_organization(db: Session, *, include_unassigned: bool = True, automatic: bool = False) -> tuple[MaintenanceJob | None, bool]:
    """Queue the organizing work that is pending. Returns (job, already_running). A background run queues
    nothing when nothing is pending, and leaves out what its last run already tried on the same state."""
    active = db.execute(select(MaintenanceJob).where(MaintenanceJob.task.in_(ORGANIZE_TASKS), MaintenanceJob.status.in_(["queued", "running"]))
                        .order_by(MaintenanceJob.id.desc()).limit(1)).scalar_one_or_none()
    if active:
        return active, True
    items = organization_items(db, include_unassigned=include_unassigned, automatic=automatic)
    if automatic:
        tried = _already_tried(db)
        items = [item for item in items if (item["kind"], item["id"], item.get("fp")) not in tried]
        if not items:
            return None, False
    row = MaintenanceJob(task=BACKGROUND_TASK if automatic else ORGANIZE_TASK, status="queued" if items else "done",
                         progress_total=len(items), items_json=json.dumps(items))
    db.add(row)
    db.flush()
    return row, False
