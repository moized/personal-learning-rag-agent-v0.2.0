"""Regression tests for the Stage 3 adversarial review."""
import json

from sqlalchemy import select

from app.config import settings
from app.models import Chunk, Concept, Evidence, MaintenanceJob, Section, Source, SourceRelationship, StudyPlan, StudyPlanNode, StudyProgress
from app.services import plans
from app.services.learning_paths import _candidate_pairs, _classify_relationships, _normalize_nodes, _source_data
from app.services.plans import PLAN_TASK
from app.services.worker import Worker
from tests.plan_jobs import run_plan_jobs
from tests.test_plans import NODE, PlanAI, _add_source, _course, _plan, env  # noqa: F401


def _build(env, course_id, regenerate=False):
    env.client.get("/syllabus", params={"course_id": course_id, **({"regenerate": "true"} if regenerate else {})})
    run_plan_jobs(env.sessions)
    return _plan(env, course_id)


def _change_content(db, source_id, concept_name):
    """Give a source's first section a new concept, so its fingerprint changes."""
    chunk = db.execute(select(Chunk).join(Section, Chunk.section_id == Section.id).where(Section.source_id == source_id).order_by(Chunk.id)).scalars().first()
    concept = Concept(canonical_name=concept_name)
    db.add(concept)
    db.flush()
    db.add(Evidence(concept_id=concept.id, chunk_id=chunk.id))


def test_a_compared_pair_whose_profiles_changed_is_recompared(env):
    with env.sessions() as db:
        course = _course(db)
        a = _add_source(db, course.id, "A", ["Gradient Descent"])
        b = _add_source(db, course.id, "B", ["Momentum"])
        db.commit()
        rows = _source_data(db, [a, b])
        _classify_relationships(db, PlanAI(relation="RELATED_BUT_SEPARATE"), rows)
        db.commit()
        b.profile_json = json.dumps({"course_title": "Optimisers"})
        db.commit()
        rows = _source_data(db, [a, b])
        _classify_relationships(db, PlanAI(relation="SAME_COURSE"), rows)
        db.commit()  # the old delete-then-insert broke the unique pair here
        assert [(r.relation_type) for r in db.execute(select(SourceRelationship)).scalars()] == ["SAME_COURSE"]


def test_a_space_whose_last_source_left_shows_no_old_topics(env):
    with env.sessions() as db:
        course = _course(db)
        source = _add_source(db, course.id, "Lectures", ["Gradient Descent"])
        db.commit()
        course_id, source_id = course.id, source.id
    assert _build(env, course_id)["nodes"]
    assert env.client.delete(f"/sources/{source_id}").status_code == 200
    run_plan_jobs(env.sessions)
    empty = _plan(env, course_id)
    assert empty["status"] == "empty" and empty["nodes"] == []
    with env.sessions() as db:
        assert db.query(StudyPlan).filter_by(scope_key=f"course:{course_id}").count() == 0


def test_progress_follows_topic_ids_when_titles_collide(env):
    with env.sessions() as db:
        course = _course(db)
        db.commit()
        course_id = course.id
    post = lambda topic, topic_id, done=True: env.client.post("/study-progress", json={"topic": topic, "topic_id": topic_id, "course_id": course_id, "completed": done})
    assert post("Alpha", "t1").status_code == 200
    # t1 was reworded to "Alpha basics" and a new topic t5 is now called "Alpha".
    assert post("Alpha", "t5", done=False).status_code == 200
    progress = env.client.get("/study-progress", params={"course_id": course_id}).json()
    assert progress["completed_topic_ids"] == ["t1"]  # t1 kept its progress
    assert post("Alpha", "t5").status_code == 200       # t5 can be completed although "Alpha" is t1's old title
    assert post("Alpha basics", "t1", done=False).status_code == 200
    assert env.client.get("/study-progress", params={"course_id": course_id}).json()["completed_topic_ids"] == ["t5"]
    assert env.client.get("/study-progress", params={"course_id": course_id}).json()["completed_legacy_topics"] == []


def test_a_previous_id_reused_in_two_paths_is_kept_once(env):
    nodes = lambda title, local: [{"id": local, "parent_id": "none", "type": "study", "title": title, "source_ids": [1], "section_ids": [], **NODE}]
    used = {"path-1", "path-2"}
    first = _normalize_nodes(nodes("Gradient Descent", "t-shared"), "path-1", {1}, {1: "x"}, frozenset({"t-shared"}), frozenset(), used)
    second = _normalize_nodes(nodes("Gradient Descent", "t-shared"), "path-2", {1}, {1: "x"}, frozenset({"t-shared"}), frozenset(), used)
    assert first[0]["node_id"] == "t-shared" and second[0]["node_id"] not in {"t-shared", "path-1", "path-2"}


def test_a_collision_gets_a_new_id_instead_of_dropping_the_topic():
    import hashlib
    momentum_id = "t-" + hashlib.sha1("path-1:momentum".encode()).hexdigest()[:12]
    raw = [{"id": momentum_id, "parent_id": "none", "type": "outline", "title": "Optimizers", "source_ids": [1], "section_ids": [], **NODE},
           {"id": "new", "parent_id": momentum_id, "type": "study", "title": "Momentum", "source_ids": [1], "section_ids": [], **NODE}]
    result = _normalize_nodes(raw, "path-1", {1}, {1: "x"}, frozenset({momentum_id}))
    assert [n["title"] for n in result] == ["Optimizers", "Momentum"] and len({n["node_id"] for n in result}) == 2


def test_quota_during_a_rebuild_keeps_the_previous_plan(env):
    with env.sessions() as db:
        course = _course(db)
        a = _add_source(db, course.id, "A", ["Gradient Descent"])
        _add_source(db, course.id, "B", ["Momentum"])
        db.commit()
        course_id, a_id = course.id, a.id
    good = _build(env, course_id)
    with env.sessions() as db:
        _change_content(db, a_id, "Learning Rate")
        db.commit()
    env.state.ai = PlanAI(quota_after=0)
    after = _build(env, course_id)
    assert after["status"] == "failed" and "quota" in after["error"]
    assert [n["title"] for n in after["nodes"]] == [n["title"] for n in good["nodes"]]  # nothing hidden
    assert env.state.ai.kinds() == ["source"]  # B's saved outline needed no call


def test_a_single_source_reworded_rebuild_keeps_topic_ids(env):
    with env.sessions() as db:
        course = _course(db)
        source = _add_source(db, course.id, "Lectures", ["Gradient Descent", "Momentum"])
        db.commit()
        course_id, source_id = course.id, source.id
    before = {n["title"]: n["id"] for n in _build(env, course_id)["nodes"] if n["type"] == "study"}

    class Reworded(PlanAI):
        def generate_json(self, prompt, task="extraction", schema=None):
            result = super().generate_json(prompt, task, schema)
            for node in result.get("nodes", []):
                node["title"] += " basics"
            return result

    env.state.ai = Reworded()
    with env.sessions() as db:
        _change_content(db, source_id, "Learning Rate")
        db.commit()
    after = {n["title"]: n["id"] for n in _build(env, course_id)["nodes"] if n["type"] == "study"}
    assert after == {f"{title} basics": node_id for title, node_id in before.items()}


def test_an_outline_without_study_topics_is_not_cached(env):
    class OutlinesOnly(PlanAI):
        def generate_json(self, prompt, task="extraction", schema=None):
            result = super().generate_json(prompt, task, schema)
            for node in result.get("nodes", []):
                node["type"] = "outline"
            return result

    env.state.ai = OutlinesOnly()
    with env.sessions() as db:
        course = _course(db)
        source = _add_source(db, course.id, "Lectures", ["Gradient Descent"])
        db.commit()
        course_id, source_id = course.id, source.id
    assert any("No study topics" in w for w in _build(env, course_id)["warnings"])
    with env.sessions() as db:
        assert db.get(Source, source_id).curriculum_json is None
    env.state.ai = PlanAI()
    assert [n["title"] for n in _build(env, course_id, regenerate=True)["nodes"] if n["type"] == "study"] == ["Gradient Descent"]


def test_a_source_embedded_by_another_model_is_compared_with_everyone(env, monkeypatch):
    with env.sessions() as db:
        course = _course(db)
        sources = [_add_source(db, course.id, f"S{i}", [f"T{i}"], vector) for i, vector in
                   enumerate([(1, 0), (0.9, 0.1), (0, 1), (0.1, 0.9)])]
        odd = _add_source(db, course.id, "Odd", ["Odd"], (1, 0, 0))  # another embedding size
        db.commit()
        pairs = _candidate_pairs(db, [s.id for s in sources] + [odd.id], 1)
    assert all(tuple(sorted((odd.id, s.id))) in pairs for s in sources)


def test_sections_beyond_the_limit_are_reported(env, monkeypatch):
    monkeypatch.setattr(settings, "plan_section_limit", 2)
    with env.sessions() as db:
        course = _course(db)
        _add_source(db, course.id, "Playlist", ["A1", "A2", "A3"])
        db.commit()
        course_id = course.id
    assert any("Only the first 2 of 3 sections of 'Playlist'" in w for w in _build(env, course_id)["warnings"])


def test_fallback_outlines_do_not_mix_topic_ids_between_sources(env):
    class LocalIdsMergeFails(PlanAI):
        def generate_json(self, prompt, task="extraction", schema=None):
            if prompt.startswith("Synthesize"):
                raise RuntimeError("merge failed")
            result = super().generate_json(prompt, task, schema)
            for index, node in enumerate(result.get("nodes", []), 1):
                node["id"] = str(index)  # real models reuse "1", "2" in every source
            if result.get("nodes"):
                result["nodes"].insert(0, {"id": "u", "parent_id": "none", "type": "outline", "title": f"Unit of {result['nodes'][0]['title']}",
                                           "source_ids": [], "section_ids": [], **NODE})
                for node in result["nodes"][1:]:
                    node["parent_id"] = "u"
            return result

    env.state.ai = LocalIdsMergeFails()
    with env.sessions() as db:
        course = _course(db)
        _add_source(db, course.id, "A", ["Gradient Descent"])
        _add_source(db, course.id, "B", ["Momentum"])
        db.commit()
        course_id = course.id
    nodes = {n["id"]: n for n in _build(env, course_id)["nodes"]}
    parent_title = lambda title: nodes[next(n for n in nodes.values() if n["title"] == title)["parent_id"]]["title"]
    assert parent_title("Gradient Descent") == "Unit of Gradient Descent" and parent_title("Momentum") == "Unit of Momentum"


def test_a_failed_refresh_is_reported(env, monkeypatch):
    with env.sessions() as db:
        course = _course(db)
        _add_source(db, course.id, "Lectures", ["Gradient Descent"])
        db.commit()
        course_id = course.id
    assert _build(env, course_id)["status"] == "ready"

    def broken(*args, **kwargs):
        raise ValueError("unexpected failure")
    monkeypatch.setattr(plans, "build_syllabus_structured", broken)
    failed = _build(env, course_id, regenerate=True)
    assert failed["status"] == "failed" and failed["nodes"]  # the old plan is still shown


def test_refresh_is_not_lost_when_the_worker_claims_the_job_first(env, monkeypatch):
    with env.sessions() as db:
        course = _course(db)
        _add_source(db, course.id, "Lectures", ["Gradient Descent"])
        plans.request_plan_update(db, course.id)
        db.commit()
        course_id = course.id
        waiting = db.query(MaintenanceJob).filter_by(task=PLAN_TASK).one()
    with env.sessions() as worker_db:  # the worker claims it in the meantime
        worker_db.get(MaintenanceJob, waiting.id).status = "running"
        worker_db.commit()
    monkeypatch.setattr(plans, "_plan_jobs", lambda db, scope, statuses=None: [db.get(MaintenanceJob, waiting.id)] if statuses == ["queued"] else [])
    with env.sessions() as db:
        job = plans.request_plan_update(db, course_id, force=True)
        db.commit()
        assert job.id != waiting.id and json.loads(job.items_json)[0]["force"] is True


def test_a_plan_build_interrupted_by_a_restart_is_requeued_at_startup(env):
    with env.sessions() as db:
        job = MaintenanceJob(task=PLAN_TASK, status="running", items_json=json.dumps([{"kind": "plan", "id": 0, "scope": "all", "course_id": None}]))
        db.add(job)
        db.commit()
        job_id = job.id
    worker = Worker()
    worker.stop_event.set()  # run the startup recovery only
    worker.run()
    with env.sessions() as db:
        assert db.get(MaintenanceJob, job_id).status == "queued"
