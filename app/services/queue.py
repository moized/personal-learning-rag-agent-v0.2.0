import json
from datetime import datetime, UTC
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from ..models import Job, Source
from .agent import initial_plan


def enqueue(db: Session, source_id: int, task: str = "process_source") -> Job:
    src = db.get(Source, source_id)
    if not src:
        raise ValueError(f"Source not found: {source_id}")
    job = Job(
        source_id=source_id,
        task=task,
        status="queued",
        stage="ingest",
        plan_json=json.dumps(initial_plan(src.source_type)),
    )
    db.add(job)
    db.flush()
    return job


def reclaim_interrupted(db: Session) -> int:
    jobs = db.execute(
        select(Job).where(Job.status.in_(["running", "retrying"]))
    ).scalars().all()
    count = 0
    for job in jobs:
        job.status = "queued"
        job.updated_at = datetime.now(UTC)
        count += 1
    db.commit()
    return count


def claim_next(db: Session) -> Job | None:
    """Claim with a compare-and-set update so two workers cannot take the same job."""
    candidate = db.execute(
        select(Job.id)
        .where(Job.status == "queued")
        .order_by(Job.id)
        .limit(1)
    ).scalar_one_or_none()
    if candidate is None:
        return None

    now = datetime.now(UTC)
    result = db.execute(
        update(Job)
        .where(Job.id == candidate, Job.status == "queued")
        .values(status="running", updated_at=now)
    )
    if result.rowcount != 1:
        db.rollback()
        return None

    db.commit()
    job = db.get(Job, candidate)
    if job:
        db.refresh(job)
    return job
