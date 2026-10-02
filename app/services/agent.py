import json
from ..models import Source, Job


def initial_plan(source_type: str) -> list[str]:
    if source_type == "youtube":
        return ["structure", "transcript", "index", "organize", "concepts", "graph"]
    return ["structure", "index", "organize", "concepts", "graph"]


def set_plan(job: Job, source: Source) -> None:
    if not job.plan_json:
        job.plan_json = json.dumps(initial_plan(source.source_type))


def next_stage(job: Job, source: Source) -> str:
    set_plan(job, source)
    plan = json.loads(job.plan_json or "[]")
    if job.status == "done":
        return "done"
    try:
        return plan[plan.index(job.stage) + 1]
    except (ValueError, IndexError):
        return plan[0] if plan else "done"
