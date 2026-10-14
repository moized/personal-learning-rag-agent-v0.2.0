"""What a student needs to know about their material, in plain words.

Internal stage names, model names, chunk ids and retrieval details stay out of these payloads.
"""
import json
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from ..config import settings
from ..models import Chunk, Decision, Document, Job, Source, Syllabus

STEP_TEXT = {
    "ingest": "Waiting to start",
    "structure": "Reading your material",
    "transcript": "Getting the video transcript",
    "dedupe": "Checking for repeated content",
    "index": "Preparing it for search",
    "organize": "Sorting it into topics",
}
PROGRESS_STAGES = {"transcript", "index", "organize"}


def _latest_jobs(db: Session, source_ids: list[int]) -> dict[int, Job]:
    jobs: dict[int, Job] = {}
    if source_ids:
        for job in db.execute(select(Job).where(Job.source_id.in_(source_ids)).order_by(Job.id)).scalars().all():
            jobs[job.source_id] = job
    return jobs


def _counts(db: Session, source_ids: list[int]) -> dict[int, tuple[int, int]]:
    """Passages kept and repeats merged per source."""
    rows = db.execute(
        select(Document.source_id, func.count(Chunk.id), func.count(Chunk.duplicate_of_id))
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.source_id.in_(source_ids)).group_by(Document.source_id)
    ).all() if source_ids else []
    return {sid: (total - merged, merged) for sid, total, merged in rows}


def _short_error(error: str | None) -> str:
    line = (error or "Something went wrong.").strip().splitlines()[0]
    return line if len(line) <= 200 else line[:197] + "..."


def source_status(src: Source, job: Job | None, counts: tuple[int, int] | None) -> dict:
    kind = {"file": "file", "transcript": "text", "youtube": "video"}.get(src.source_type, "file")
    out = {"id": src.id, "name": src.name, "kind": kind, "progress": None, "summary": None, "error": None,
           "decision_id": job.waiting_decision_id if job else None}
    status = job.status if job else src.status
    if status == "done":
        passages, merged = counts or (0, 0)
        summary = f"{passages} passage{'s' if passages != 1 else ''}"
        if merged:
            summary += f", {merged} repeated passage{'s' if merged != 1 else ''} merged"
        return {**out, "state": "ready", "step": "Ready", "summary": summary}
    if status == "failed":
        return {**out, "state": "failed", "step": "Could not be processed", "error": _short_error(job.error if job else src.error)}
    if status == "waiting":
        return {**out, "state": "needs_help", "step": "Waiting for your AI's answer"}
    if status == "queued" and job and job.retry_count:
        return {**out, "state": "working",
                "step": f"Hit a problem, trying again (attempt {job.retry_count + 1} of {settings.retry_max + 1})"}
    if status == "queued":
        return {**out, "state": "queued", "step": "Waiting in line"}
    stage = job.stage if job else "ingest"
    if job and stage in PROGRESS_STAGES and job.progress_total:
        out["progress"] = round(min(1.0, job.progress_current / job.progress_total), 3)
    return {**out, "state": "working", "step": STEP_TEXT.get(stage, "Working on it")}


def course_sources(db: Session, course_id: int) -> list[dict]:
    sources = db.execute(select(Source).where(Source.course_id == course_id).order_by(Source.id)).scalars().all()
    ids = [s.id for s in sources]
    jobs, counts = _latest_jobs(db, ids), _counts(db, ids)
    return [source_status(s, jobs.get(s.id), counts.get(s.id)) for s in sources]


def plan_outdated(db: Session, syllabus: Syllabus) -> bool:
    """True when material finished processing after the plan was made."""
    cond = Source.course_id == syllabus.course_id if syllabus.course_id is not None else Source.course_id.is_(None)
    newest = db.execute(select(func.max(Source.updated_at)).where(cond, Source.status == "done")).scalar_one_or_none()
    return bool(newest and syllabus.created_at and newest.replace(tzinfo=None) > syllabus.created_at.replace(tzinfo=None))


def decision_context(db: Session, decision: Decision) -> dict:
    """Which course (and source) a decision belongs to, so the page can show it in the right place."""
    if decision.job_id:
        job = db.get(Job, decision.job_id)
        src = db.get(Source, job.source_id) if job else None
        if src:
            return {"course_id": src.course_id, "source_id": src.id}
    args = json.loads(decision.action_args_json or "{}")
    return {"course_id": args.get("course_id"), "source_id": None}
