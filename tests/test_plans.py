"""Stage 3: study plans built in the background, cached by content, read without AI calls."""
import json
import re
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models import Base, Chunk, Concept, Course, Document, Evidence, MaintenanceJob, Source, StudyPlan, StudyPlanNode
from app.providers.base import ProviderQuotaExceededError
from app.services import plans, worker as worker_module
from app.services.plans import PLAN_TASK, plan_revision
from app.services.sections import build_source_sections
from tests.plan_jobs import run_plan_jobs

NODE = {"purpose": "p", "prerequisites": "", "prerequisite_confidence": 0, "why_included": "", "enables": "", "source_roles": [],
        "topic_merge_confidence": 0.5, "core_or_supplementary": "core", "comes_after": []}


class PlanAI:
    """Answers the plan builder's three kinds of request from what the prompt contains."""

    def __init__(self, relation="SAME_COURSE", rename=None, quota_after=None):
        self.prompts, self.relation, self.rename, self.quota_after = [], relation, rename or {}, quota_after

    def kinds(self):
        return [kind for kind, _ in self.prompts]

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def generate_json(self, prompt, task="extraction", schema=None):
        if prompt.startswith("Extract the learning curriculum"):
            self.prompts.append(("source", prompt))
            if self.quota_after is not None and len([k for k in self.kinds() if k == "source"]) > self.quota_after:
                raise ProviderQuotaExceededError("daily quota exhausted")
            sid = int(re.search(r"source_ids \[(\d+)\]", prompt).group(1))
            sections = json.loads(re.search(r"short excerpt\)(?: \(.*?\))?: (\[.*\])\nReturn nodes", prompt, re.S).group(1))
            nodes = [{"id": f"S{s['id']}", "parent_id": "none", "type": "study", "title": s["title"], "source_ids": [sid],
                      "section_ids": [s["id"], 999999], **NODE} for s in sections]
            return {"nodes": nodes, "knowledge": [], "skills": [], "capabilities": [], "applications": []}
        if prompt.startswith("Classify these source pairs"):
            self.prompts.append(("compare", prompt))
            pairs = json.loads(re.search(r"Requested pairs: (\[.*\])", prompt).group(1))
            return {"relationships": [{"source_a_id": a, "source_b_id": b, "type": self.relation, "confidence": 0.9,
                                       "rationale": "r", "evidence": []} for a, b in pairs]}
        if prompt.startswith("Synthesize these independent"):
            self.prompts.append(("merge", prompt))
            inputs = json.loads(re.search(r"relationship evidence: (.*)\nSource identity classifications", prompt, re.S).group(1))
            paths = []
            for path in inputs:
                nodes = [dict(n) for c in path["source_curricula"] for n in c.get("nodes", [])]
                for node in nodes:
                    if node["title"] in self.rename:
                        node["id"], node["title"] = self.rename[node["title"]]
                paths.append({"path_id": path["path_id"], "title": path["suggested_title"], "source_ids": path["source_ids"],
                              "primary_goal": "g", "knowledge": [], "skills": [], "capabilities": [], "applications": [], "nodes": nodes})
            return {"paths": paths, "cross_path_relations": [], "shared_foundations": []}
        return {}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'plans.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    @contextmanager
    def scope():
        db = sessions()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    from app import main
    monkeypatch.setattr(main, "session_scope", scope)
    monkeypatch.setattr(worker_module, "session_scope", scope)
    state = SimpleNamespace(ai=PlanAI())
    monkeypatch.setattr(worker_module, "provider", lambda: state.ai)
    yield SimpleNamespace(sessions=sessions, client=TestClient(main.app), state=state)
    engine.dispose()


def _add_source(db, course_id, name, topics, vector=(1.0, 0.0)):
    source = Source(name=name, source_type="youtube", status="done", course_id=course_id)
    db.add(source)
    db.flush()
    for index, topic in enumerate(topics):
        doc = Document(source_id=source.id, external_id=f"{name}-{index}", title=topic)
        db.add(doc)
        db.flush()
        chunk = Chunk(document_id=doc.id, chunk_index=0, chunk_kind="leaf", text=f"{topic} explained.", embedding_status="done",
                      embedding_json=json.dumps(list(vector)))
        db.add(chunk)
        db.flush()
        concept = db.execute(select(Concept).where(Concept.canonical_name == topic)).scalar_one_or_none() or Concept(canonical_name=topic)
        db.add(concept)
        db.flush()
        db.add(Evidence(concept_id=concept.id, chunk_id=chunk.id))
    db.flush()
    build_source_sections(db, source)
    return source


def _course(db, name="Deep Learning"):
    course = Course(name=name)
    db.add(course)
    db.flush()
    return course


def _plan(env, course_id):
    return env.client.get("/syllabus", params={"course_id": course_id}).json()


def test_opening_a_plan_never_calls_the_ai_and_requests_are_merged(env):
    class NoAI(PlanAI):
        def generate_json(self, *args, **kwargs):
            raise AssertionError("the AI must not be called while reading a plan")
    env.state.ai = NoAI()
    with env.sessions() as db:
        course = _course(db)
        _add_source(db, course.id, "Lectures", ["Gradient Descent"])
        db.commit()
        course_id = course.id
    assert _plan(env, course_id)["status"] == "updating"
    assert _plan(env, course_id)["status"] == "updating"
    with env.sessions() as db:
        assert db.query(MaintenanceJob).filter_by(task=PLAN_TASK).count() == 1


def test_adding_a_source_only_reads_the_new_source(env):
    with env.sessions() as db:
        course = _course(db)
        _add_source(db, course.id, "Lectures", ["Gradient Descent", "Backpropagation"])
        _add_source(db, course.id, "Notes", ["Learning Rate"])
        db.commit()
        course_id = course.id
    _plan(env, course_id)
    run_plan_jobs(env.sessions)
    first = _plan(env, course_id)
    assert first["status"] == "ready" and env.state.ai.kinds().count("source") == 2
    with env.sessions() as db:
        _add_source(db, course_id, "Book", ["Momentum"])
        db.commit()
    assert _plan(env, course_id)["status"] == "updating"
    env.state.ai.prompts.clear()
    run_plan_jobs(env.sessions)
    sources = [p for kind, p in env.state.ai.prompts if kind == "source"]
    assert len(sources) == 1 and "Momentum" in sources[0]  # the other two outlines were reused
    titles = {n["title"] for n in _plan(env, course_id)["nodes"] if n["type"] == "study"}
    assert {"Gradient Descent", "Backpropagation", "Learning Rate", "Momentum"} <= titles


def test_non_content_updates_do_not_outdate_the_plan(env):
    with env.sessions() as db:
        course = _course(db)
        source = _add_source(db, course.id, "Lectures", ["Gradient Descent"])
        db.commit()
        course_id, source_id = course.id, source.id
    _plan(env, course_id)
    run_plan_jobs(env.sessions)
    with env.sessions() as db:
        source = db.get(Source, source_id)
        source.assignment_reason = "Changed explicitly in the library."
        source.error = None
        db.commit()
    assert _plan(env, course_id)["status"] == "ready"
    with env.sessions() as db:
        _add_source(db, course_id, "More", ["Momentum"])
        db.commit()
    assert _plan(env, course_id)["status"] == "updating"


def test_a_single_source_needs_no_merge_call(env):
    with env.sessions() as db:
        course = _course(db)
        _add_source(db, course.id, "Lectures", ["Gradient Descent", "Momentum"])
        db.commit()
        course_id = course.id
    _plan(env, course_id)
    run_plan_jobs(env.sessions)
    assert env.state.ai.kinds() == ["source"]
    nodes = [n for n in _plan(env, course_id)["nodes"] if n["type"] == "study"]
    with env.sessions() as db:
        section_ids = set(db.execute(select(Chunk.section_id)).scalars())
    assert nodes and all(set(n["metadata"]["section_ids"]) <= section_ids and n["metadata"]["section_ids"] for n in nodes)


def test_topic_ids_and_progress_survive_a_reworded_rebuild(env):
    with env.sessions() as db:
        course = _course(db)
        _add_source(db, course.id, "Lectures", ["Gradient Descent"])
        _add_source(db, course.id, "Notes", ["Learning Rate"])
        db.commit()
        course_id = course.id
    _plan(env, course_id)
    run_plan_jobs(env.sessions)
    topic = next(n for n in _plan(env, course_id)["nodes"] if n["title"] == "Gradient Descent")
    assert env.client.post("/study-progress", json={"topic": topic["title"], "topic_id": topic["id"], "course_id": course_id}).status_code == 200

    env.state.ai = PlanAI(rename={"Gradient Descent": (topic["id"], "Gradient Descent Basics")})
    env.client.get("/syllabus", params={"course_id": course_id, "regenerate": "true"})
    run_plan_jobs(env.sessions)
    renamed = next(n for n in _plan(env, course_id)["nodes"] if n["id"] == topic["id"])
    assert renamed["title"] == "Gradient Descent Basics"
    assert topic["id"] in env.state.ai.prompts[-1][1]  # the merge was shown the previous outline
    progress = env.client.get("/study-progress", params={"course_id": course_id}).json()
    assert topic["id"] in progress["completed_topic_ids"]
    env.client.post("/study-progress", json={"topic": "Gradient Descent Basics", "topic_id": topic["id"], "course_id": course_id, "completed": False})
    assert env.client.get("/study-progress", params={"course_id": course_id}).json()["completed_topic_ids"] == []


def test_only_similar_sources_are_compared(env, monkeypatch):
    monkeypatch.setattr(settings, "plan_compare_k", 1)
    with env.sessions() as db:
        course = _course(db)
        vectors = [(1, 0), (0.99, 0.1), (0, 1), (0.1, 0.99), (-1, 0), (-0.99, -0.1)]
        for index, vector in enumerate(vectors):
            _add_source(db, course.id, f"Source {index}", [f"Topic {index}"], vector)
        db.commit()
        course_id = course.id
    env.state.ai = PlanAI(relation="RELATED_BUT_SEPARATE")
    _plan(env, course_id)
    run_plan_jobs(env.sessions)
    compare = next(p for kind, p in env.state.ai.prompts if kind == "compare")
    pairs = json.loads(re.search(r"Requested pairs: (\[.*\])", compare).group(1))
    assert len(pairs) == 3  # the three similar pairs, not all 15


def test_a_failed_build_is_not_retried_in_a_loop(env, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("unexpected failure")
    monkeypatch.setattr(plans, "build_syllabus_structured", broken)
    with env.sessions() as db:
        course = _course(db)
        _add_source(db, course.id, "Lectures", ["Gradient Descent"])
        db.commit()
        course_id = course.id
    _plan(env, course_id)
    run_plan_jobs(env.sessions)
    failed = _plan(env, course_id)
    assert failed["status"] == "failed" and "Refresh" in failed["error"]
    assert _plan(env, course_id)["status"] == "failed" and run_plan_jobs(env.sessions) == 0  # no automatic retry
    assert env.client.get("/syllabus", params={"course_id": course_id, "regenerate": "true"}).json()["status"] == "updating"


def test_daily_quota_stops_further_calls(env):
    env.state.ai = PlanAI(relation="RELATED_BUT_SEPARATE", quota_after=1)
    with env.sessions() as db:
        course = _course(db)
        for index in range(3):
            _add_source(db, course.id, f"Source {index}", [f"Topic {index}"])
        db.commit()
        course_id = course.id
    _plan(env, course_id)
    run_plan_jobs(env.sessions)
    assert env.state.ai.kinds().count("source") == 2 and "merge" not in env.state.ai.kinds()
    assert any("daily quota" in w for w in _plan(env, course_id)["warnings"])


def test_source_changes_queue_plan_updates_in_the_background(env, monkeypatch):
    with env.sessions() as db:
        first, second = _course(db, "A"), _course(db, "B")
        source = _add_source(db, first.id, "Lectures", ["Gradient Descent"])
        db.add(StudyPlan(scope_key="all", revision="old", syllabus="x"))
        db.commit()
        source_id, first_id, second_id = source.id, first.id, second.id
    env.client.patch(f"/sources/{source_id}/course", json={"course_id": second_id})
    with env.sessions() as db:
        scopes = sorted(json.loads(j.items_json)[0]["scope"] for j in db.query(MaintenanceJob).filter_by(task=PLAN_TASK))
        assert scopes == ["all", f"course:{first_id}", f"course:{second_id}"]
        db.query(MaintenanceJob).delete()
        db.commit()
    assert env.client.delete(f"/sources/{source_id}").status_code == 200
    with env.sessions() as db:
        scopes = sorted(json.loads(j.items_json)[0]["scope"] for j in db.query(MaintenanceJob).filter_by(task=PLAN_TASK))
        assert scopes == ["all", f"course:{second_id}"]


def test_a_finished_source_queues_its_plan(env, monkeypatch, tmp_path):
    note = tmp_path / "notes.md"
    note.write_text("# Gradient descent\nIt lowers the loss.", encoding="utf-8")
    with env.sessions() as db:
        course = _course(db)
        source = Source(name="notes.md", source_type="file", local_path=str(note), status="queued", course_id=course.id,
                        assignment_mode="explicit_course")
        db.add(source)
        db.flush()
        from app.services.queue import enqueue
        job = enqueue(db, source.id)
        db.commit()
        job_id, course_id = job.id, course.id
    worker_module.Worker().process_job(job_id)
    with env.sessions() as db:
        assert [json.loads(j.items_json)[0]["scope"] for j in db.query(MaintenanceJob).filter_by(task=PLAN_TASK)] == [f"course:{course_id}"]


def test_sources_from_before_sections_still_get_a_plan(env):
    with env.sessions() as db:
        course = _course(db)
        source = Source(name="Old lecture", source_type="transcript", status="done", course_id=course.id)
        db.add(source)
        db.flush()
        doc = Document(source_id=source.id, external_id="old", title="Old lecture")
        db.add(doc)
        db.flush()
        db.add(Chunk(document_id=doc.id, chunk_index=0, chunk_kind="leaf", text="Momentum smooths updates."))
        db.commit()
        course_id = course.id
    assert plan_revision(env.sessions(), course_id) is not None
    _plan(env, course_id)
    run_plan_jobs(env.sessions)
    assert [n["title"] for n in _plan(env, course_id)["nodes"] if n["type"] == "study"] == ["Old lecture"]
