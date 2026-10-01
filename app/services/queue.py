import json
from datetime import datetime, UTC
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..models import Job, Source
from .agent import initial_plan


def enqueue(db: Session, source_id: int, task: str = "process_source") -> Job:
    src = db.get(Source, source_id)
    job = Job(source_id=source_id, task=task, status="queued", stage="ingest", plan_json=json.dumps(initial_plan(src.source_type)))
    db.add(job); db.flush()
    return job


def reclaim_interrupted(db: Session) -> int:
    jobs = db.execute(select(Job).where(Job.status.in_(["running", "retrying"]))).scalars().all()
    count = 0
    for job in jobs:
        job.status = "queued"
        job.updated_at = datetime.now(UTC)
        count += 1
    db.commit()
    return count


def claim_next(db: Session) -> Job | None:
    job = db.execute(select(Job).where(Job.status == "queued").order_by(Job.id)).scalars().first()
    if not job:
        return None
    job.status = "running"
    job.updated_at = datetime.now(UTC)
    db.commit(); db.refresh(job)
    return job
