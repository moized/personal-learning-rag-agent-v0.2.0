"""Test helper: run queued study-plan builds the way the background worker does."""
from app.models import MaintenanceJob
from app.services.plans import PLAN_TASK
from app.services.worker import Worker


def run_plan_jobs(sessions) -> int:
    """Process every queued plan job. The caller must have patched the worker's session_scope."""
    with sessions() as db:
        job_ids = [job.id for job in db.query(MaintenanceJob).filter_by(task=PLAN_TASK, status="queued").order_by(MaintenanceJob.id)]
        for job_id in job_ids:
            db.get(MaintenanceJob, job_id).status = "running"  # what claiming does
        db.commit()
    for job_id in job_ids:
        Worker()._process_maintenance_job(job_id)
    return len(job_ids)
