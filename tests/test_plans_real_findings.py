"""Regression tests for bugs found by running Stage 3 against real Gemini."""
import json

from app.models import MaintenanceJob
from app.services import plans
from app.services.plans import PLAN_TASK
from tests.plan_jobs import run_plan_jobs
from tests.test_plans import PlanAI, _add_source, _course, _plan, env  # noqa: F401


def test_a_topic_cannot_take_the_path_roots_id(env):
    with env.sessions() as db:
        course = _course(db)
        _add_source(db, course.id, "Lectures", ["Gradient Descent"])
        _add_source(db, course.id, "Notes", ["Learning Rate"])
        db.commit()
        course_id = course.id
    _plan(env, course_id)
    run_plan_jobs(env.sessions)
    root = next(n for n in _plan(env, course_id)["nodes"] if n["parent_id"] is None)
    # A real model returned the root's id ("path-1") for a topic when shown the previous outline.
    env.state.ai = PlanAI(rename={"Gradient Descent": (root["id"], "Gradient Descent")})
    env.client.get("/syllabus", params={"course_id": course_id, "regenerate": "true"})
    run_plan_jobs(env.sessions)
    rebuilt = _plan(env, course_id)
    assert rebuilt["status"] == "ready"
    ids = [n["id"] for n in rebuilt["nodes"]]
    assert len(ids) == len(set(ids)) and "Gradient Descent" in [n["title"] for n in rebuilt["nodes"]]
    offered = json.loads(env.state.ai.prompts[-1][1].split("progress; drop topics whose sources are gone: ")[1])
    assert root["id"] not in [entry["id"] for entry in offered]  # roots are not offered for reuse (they may be parents)


def test_a_failed_background_build_without_a_revision_is_not_retried(env, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("unexpected failure")
    monkeypatch.setattr(plans, "build_syllabus_structured", broken)
    with env.sessions() as db:
        course = _course(db)
        _add_source(db, course.id, "Lectures", ["Gradient Descent"])
        plans.request_updates_after_change(db, {course.id})  # what a finished source queues: no revision recorded
        db.commit()
        course_id = course.id
    run_plan_jobs(env.sessions)
    assert _plan(env, course_id)["status"] == "failed"
    assert _plan(env, course_id)["status"] == "failed"
    with env.sessions() as db:
        assert db.query(MaintenanceJob).filter_by(task=PLAN_TASK, status="queued").count() == 0
        job = db.query(MaintenanceJob).filter_by(task=PLAN_TASK).one()
        assert json.loads(job.items_json)[0]["revision"]  # stamped by the worker before building


def test_rebuilding_unchanged_sections_keeps_their_ids_and_the_plan_up_to_date(env):
    from sqlalchemy import select
    from app.models import Section, Source
    from app.services.sections import build_source_sections
    with env.sessions() as db:
        course = _course(db)
        source = _add_source(db, course.id, "Lectures", ["Gradient Descent", "Momentum"])
        db.commit()
        course_id, source_id = course.id, source.id
    _plan(env, course_id)
    run_plan_jobs(env.sessions)
    with env.sessions() as db:
        before = db.execute(select(Section.id).where(Section.source_id == source_id).order_by(Section.position)).scalars().all()
        build_source_sections(db, db.get(Source, source_id))  # what every re-index or backfill does
        db.commit()
        after = db.execute(select(Section.id).where(Section.source_id == source_id).order_by(Section.position)).scalars().all()
    assert before == after
    assert _plan(env, course_id)["status"] == "ready"
