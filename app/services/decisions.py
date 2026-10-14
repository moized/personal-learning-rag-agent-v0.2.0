"""Paste-back decisions: the system asks, the user answers with an external AI, the system resumes."""
import json
import re
from datetime import datetime, UTC
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..models import Decision, Job, Source
from ..providers.base import DecisionPending
from .ingestion import parse_structured_literal

FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def record_decision(db: Session, pending: DecisionPending, *, job_id: int | None = None,
                    action: str | None = None, action_args: dict | None = None) -> Decision:
    """Create or refresh the Decision row for a DecisionPending raised by the manual provider."""
    decision = db.execute(select(Decision).where(Decision.key == pending.key)).scalar_one_or_none()
    if decision is None:
        decision = Decision(key=pending.key, task=pending.task, kind=pending.kind, prompt=pending.prompt)
        db.add(decision)
    if decision.status != "answered":
        # A caller-supplied key keeps its identity while the prompt is rebuilt from newer state.
        decision.prompt = pending.prompt
        decision.status = "pending"
    decision.label = pending.label
    decision.schema_json = json.dumps(pending.schema) if pending.schema else None
    if job_id is not None:
        decision.job_id = job_id
    if action is not None:
        decision.action = action
        decision.action_args_json = json.dumps(action_args or {})
        db.flush()
        for stale in db.execute(select(Decision).where(
            Decision.action == action, Decision.action_args_json == decision.action_args_json,
            Decision.status == "pending", Decision.id != decision.id,
        )).scalars().all():
            stale.status = "superseded"
    db.flush()
    return decision


def pause_job(db: Session, job_id: int, pending: DecisionPending) -> Decision:
    decision = record_decision(db, pending, job_id=job_id)
    job = db.get(Job, job_id)
    if job.waiting_decision_id and job.waiting_decision_id != decision.id:
        previous = db.get(Decision, job.waiting_decision_id)
        if previous and previous.status == "pending":
            previous.status = "superseded"
    job.status = "waiting"
    job.waiting_decision_id = decision.id
    job.error = None
    src = db.get(Source, job.source_id)
    if src:
        src.status = "waiting"
    db.flush()
    return decision


def parse_json_answer(text: str, schema: dict | None) -> dict:
    raw = FENCE_RE.sub("", text.strip()).strip()
    try:
        obj = parse_structured_literal(raw)
    except ValueError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The answer does not contain a JSON object")
        obj = parse_structured_literal(raw[start:end + 1])
    missing = [k for k in (schema or {}).get("required", []) if k not in obj]
    if missing:
        raise ValueError(f"The answer is missing required field(s): {', '.join(missing)}")
    return obj


def submit_answer(db: Session, decision_id: int, answer: str) -> Decision:
    decision = db.get(Decision, decision_id)
    if decision is None:
        raise LookupError("Decision not found")
    if not answer.strip():
        raise ValueError("The answer is empty")
    if decision.kind == "json":
        schema = json.loads(decision.schema_json) if decision.schema_json else None
        decision.result_json = json.dumps(parse_json_answer(answer, schema), ensure_ascii=False)
    decision.answer = answer.strip()
    decision.status = "answered"
    decision.answered_at = datetime.now(UTC)
    db.flush()
    return decision


def requeue_waiting_jobs(db: Session, decision_id: int | None = None) -> list[int]:
    """Re-queue jobs paused on an answered decision (one decision, or all after a restart)."""
    stmt = select(Job).where(Job.status == "waiting")
    if decision_id is not None:
        stmt = stmt.where(Job.waiting_decision_id == decision_id)
    requeued = []
    for job in db.execute(stmt).scalars().all():
        decision = db.get(Decision, job.waiting_decision_id) if job.waiting_decision_id else None
        if decision is not None and decision.status == "pending":
            continue
        job.status = "queued"
        job.waiting_decision_id = None
        src = db.get(Source, job.source_id)
        if src:
            src.status = "queued"
        requeued.append(job.id)
    return requeued


def serialize(decision: Decision, with_prompt: bool = True) -> dict:
    out = {
        "id": decision.id,
        "task": decision.task,
        "kind": decision.kind,
        "label": decision.label,
        "status": decision.status,
        "job_id": decision.job_id,
        "action": decision.action,
        "created_at": decision.created_at.isoformat() if decision.created_at else None,
    }
    if with_prompt:
        out["prompt"] = decision.prompt
    return out
