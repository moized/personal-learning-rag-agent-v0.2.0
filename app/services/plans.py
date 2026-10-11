"""Study plans: built in the background, cached by content, read without AI calls.

A plan belongs to a scope ("course:<id>" or "all"). Its revision is a fingerprint of what
it describes: each source's profile, sections and concepts. Opening a plan only compares
revisions; building happens in a maintenance job queued per scope (at most one waiting at
a time), triggered when sources change or when a reader finds the plan outdated.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models import Course, MaintenanceJob, Source, StudyPlan, StudyPlanNode
from ..providers.base import AIProvider
from .curriculum import build_syllabus_structured
from .learning_paths import _source_data, source_fingerprint

PLAN_TASK = "update_plan"
NO_SOURCES = "No completed sources are available in this scope yet."


def scope_key(course_id: int | None) -> str:
    return f"course:{course_id}" if course_id is not None else "all"


def scope_sources(db: Session, course_id: int | None) -> list[Source]:
    query = select(Source).where(Source.status == "done").order_by(Source.id)
    if course_id is not None:
        query = query.where(Source.course_id == course_id)
    return db.execute(query).scalars().all()


def plan_revision(db: Session, course_id: int | None) -> str | None:
    """Fingerprint of what the plan for this scope describes; None when it has no sources."""
    sources = scope_sources(db, course_id)
    if not sources:
        return None
    parts: list = [[item["id"], source_fingerprint(item)] for item in _source_data(db, sources)]
    if course_id is not None:
        course = db.get(Course, course_id)
        if course is not None:
            parts.append(["course", course.name, course.description or "", course.outline or ""])
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def _item(job: MaintenanceJob) -> dict:
    try:
        items = json.loads(job.items_json or "[]")
    except json.JSONDecodeError:
        return {}
    return items[0] if items and isinstance(items[0], dict) else {}


def _plan_jobs(db: Session, scope: str, statuses: list[str] | None = None) -> list[MaintenanceJob]:
    query = select(MaintenanceJob).where(MaintenanceJob.task == PLAN_TASK).order_by(MaintenanceJob.id.desc())
    if statuses:
        query = query.where(MaintenanceJob.status.in_(statuses))
    return [job for job in db.execute(query).scalars().all() if _item(job).get("scope") == scope]


def active_plan_job(db: Session, scope: str) -> MaintenanceJob | None:
    jobs = _plan_jobs(db, scope, ["queued", "running"])
    return jobs[0] if jobs else None


def latest_plan_job(db: Session, scope: str) -> MaintenanceJob | None:
    jobs = _plan_jobs(db, scope)
    return jobs[0] if jobs else None


def request_plan_update(db: Session, course_id: int | None, *, force: bool = False, revision: str | None = None) -> MaintenanceJob:
    """Queue a background rebuild for a scope. At most one waits per scope: a second request
    joins it (a forced request upgrades it). A running build gets one follow-up queued, so
    changes made while it runs are picked up."""
    scope = scope_key(course_id)
    for job in _plan_jobs(db, scope, ["queued"]):
        if not force or _item(job).get("force"):
            return job
        upgraded = db.execute(update(MaintenanceJob).where(MaintenanceJob.id == job.id, MaintenanceJob.status == "queued")
                              .values(items_json=json.dumps([{**_item(job), "force": True}]))).rowcount
        if upgraded:
            return job
        break  # the worker claimed it meanwhile: queue a new forced build
    job = MaintenanceJob(task=PLAN_TASK, status="queued", progress_total=1, items_json=json.dumps([{
        "kind": "plan", "id": course_id if course_id is not None else 0, "scope": scope,
        "course_id": course_id, "force": force, "revision": revision,
    }]))
    db.add(job)
    db.flush()
    return job


def request_updates_after_change(db: Session, course_ids: set[int | None]) -> None:
    """Sources in these courses changed: refresh their plans, and the all-sources plan if one exists."""
    for course_id in sorted({c for c in course_ids if c is not None}):
        request_plan_update(db, course_id)
    if db.execute(select(StudyPlan.id).where(StudyPlan.scope_key == "all")).first():
        request_plan_update(db, None)


def build_and_store_plan(db: Session, ai: AIProvider, course_id: int | None, *, force: bool = False, heartbeat=None) -> StudyPlan | None:
    """Rebuild a scope's plan when its content changed (or when forced) and save it."""
    scope = scope_key(course_id)
    revision = plan_revision(db, course_id)
    plan = db.execute(select(StudyPlan).where(StudyPlan.scope_key == scope)).scalar_one_or_none()
    if revision is None:
        if plan is not None:  # its last source was removed or moved: its topics are no longer valid
            db.query(StudyPlanNode).filter(StudyPlanNode.plan_id == plan.id).delete(synchronize_session=False)
            db.delete(plan)
            db.flush()
        return None
    if plan is not None and plan.revision == revision and not force:
        return plan
    previous = [{"id": node.node_id, "title": node.title, "parent_id": node.parent_node_id, "type": node.node_type,
                 "section_ids": json.loads(node.metadata_json or "{}").get("section_ids") or []}
                for node in _nodes(db, plan)] if plan is not None else None
    rendered, structured_nodes = build_syllabus_structured(db, ai, course_id, previous_nodes=previous, heartbeat=heartbeat)
    relationships_json = json.dumps({
        "sources": db.info.get("syllabus_source_relationships", []),
        "cross_paths": db.info.get("syllabus_cross_path_relations", []),
        "shared_foundations": db.info.get("syllabus_shared_foundations", []),
        "warnings": db.info.get("syllabus_warnings", []),
    }, ensure_ascii=False)
    if plan is None:
        plan = StudyPlan(scope_key=scope, revision=revision, syllabus=rendered, relationships_json=relationships_json)
        db.add(plan)
        try:
            db.flush()
        except IntegrityError:
            # Another process saved this scope first; overwrite it with this newer build.
            db.rollback()
            plan = db.execute(select(StudyPlan).where(StudyPlan.scope_key == scope)).scalar_one()
    plan.revision = revision
    plan.syllabus = rendered
    plan.relationships_json = relationships_json
    plan.generated_at = datetime.now(UTC)
    db.query(StudyPlanNode).filter(StudyPlanNode.plan_id == plan.id).delete(synchronize_session=False)
    for node_data in structured_nodes:
        db.add(StudyPlanNode(plan_id=plan.id, **node_data))
    db.flush()
    return plan


def _nodes(db: Session, plan: StudyPlan | None) -> list[StudyPlanNode]:
    if plan is None:
        return []
    return db.execute(select(StudyPlanNode).where(StudyPlanNode.plan_id == plan.id).order_by(StudyPlanNode.id)).scalars().all()


def node_json(node: StudyPlanNode) -> dict:
    return {"id": node.node_id, "parent_id": node.parent_node_id, "path_id": node.path_id, "type": node.node_type,
            "title": node.title, "metadata": json.loads(node.metadata_json or "{}")}


def plan_payload(db: Session, plan: StudyPlan | None) -> dict:
    """The saved plan in API form, tolerating older relationship formats."""
    if plan is None:
        return {"syllabus": "", "revision": None, "generated_at": None, "nodes": [],
                "source_relationships": {"sources": [], "cross_paths": [], "shared_foundations": []}, "warnings": []}
    try:
        saved = json.loads(plan.relationships_json or "{}")
    except (TypeError, json.JSONDecodeError):
        saved = {}
    if not isinstance(saved, dict):
        saved = {"sources": saved if isinstance(saved, list) else []}
    relationships = {key: saved.get(key) if isinstance(saved.get(key), list) else [] for key in ("sources", "cross_paths", "shared_foundations")}
    warnings = [w for w in saved["warnings"] if isinstance(w, str)] if isinstance(saved.get("warnings"), list) else []
    return {"syllabus": plan.syllabus, "revision": plan.revision, "generated_at": plan.generated_at.isoformat() if plan.generated_at else None,
            "nodes": [node_json(node) for node in _nodes(db, plan)], "source_relationships": relationships,
            # Parts of the plan that could not be built (for example the AI service was
            # unavailable). The plan is still saved so it can be shown; Refresh rebuilds it.
            "warnings": warnings}


def read_plan(db: Session, course_id: int | None, *, regenerate: bool = False) -> dict:
    """The saved plan for a scope plus its status. Never calls the AI.

    Status: "empty" (no sources), "ready" (up to date), "updating" (a build is queued or
    running), "outdated" (content changed, nothing queued), "failed" (the last build for this
    content failed; it is not retried automatically, so a quota problem does not loop).
    """
    scope = scope_key(course_id)
    revision = plan_revision(db, course_id)
    plan = db.execute(select(StudyPlan).where(StudyPlan.scope_key == scope)).scalar_one_or_none()
    job = active_plan_job(db, scope)
    last = latest_plan_job(db, scope)
    failed_here = last is not None and last.status == "failed" and _item(last).get("revision") == revision
    if revision is not None:
        if regenerate:
            job = request_plan_update(db, course_id, force=True, revision=revision)
        elif job is None and (plan is None or plan.revision != revision) and not failed_here:
            job = request_plan_update(db, course_id, revision=revision)
    up_to_date = plan is not None and plan.revision == revision
    # A failed forced Refresh leaves an up-to-date plan in place; the reader must still hear about it.
    failed_after_plan = (last is not None and last.status == "failed" and plan is not None and last.updated_at is not None
                         and plan.generated_at is not None and last.updated_at.replace(tzinfo=None) > plan.generated_at.replace(tzinfo=None))
    if revision is None:
        status = "empty"
    elif job is not None:
        status = "updating"
    elif (failed_here and not up_to_date) or failed_after_plan:
        status = "failed"
    elif up_to_date:
        status = "ready"
    else:
        status = "outdated"
    payload = plan_payload(db, plan if revision is not None else None)
    if revision is None:
        payload["syllabus"] = NO_SOURCES
    payload.update({"status": status, "up_to_date": up_to_date, "error": None})
    if status == "failed":
        try:
            errors = json.loads(last.errors_json or "{}")
        except json.JSONDecodeError:
            errors = {}
        payload["error"] = "The last update of this plan failed. Press Refresh to try again." + (
            f" ({next(iter(errors.values()))[:200]})" if isinstance(errors, dict) and errors else "")
    return payload
