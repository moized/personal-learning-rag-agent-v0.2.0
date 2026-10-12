from contextlib import contextmanager
import json

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Chunk, Course, Document, Job, Source, SourceRelationship, StudyPlan, StudyPlanNode, MaintenanceJob
from app.services import worker as worker_module
from app.services.worker import Worker
from app.services.learning_paths import _classify_relationships, group_source_paths


def _scope(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'architecture.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)

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

    monkeypatch.setattr(worker_module, "session_scope", scope)
    from app import main
    monkeypatch.setattr(main, "session_scope", scope)
    return TestClient(main.app), sessions, engine


def test_assignment_intent_is_persisted_and_duplicate_conflict_is_visible(tmp_path, monkeypatch):
    client, sessions, engine = _scope(tmp_path, monkeypatch)
    space = client.post("/courses", json={"name": "Course A"}).json()
    auto = client.post("/sources/youtube", params={"url": "https://youtu.be/BaW_jenozKc", "assignment_mode": "auto"}).json()
    fixed = client.post("/sources/youtube", params={"url": "https://youtu.be/aqz-KE-bpKQ", "course_id": space["id"], "assignment_mode": "explicit_course"}).json()
    duplicate = client.post("/sources/youtube", params={"url": "https://www.youtube.com/watch?v=aqz-KE-bpKQ", "assignment_mode": "explicit_unassigned"}).json()
    with sessions() as db:
        assert db.get(Source, auto["source_id"]).assignment_mode == "auto"
        fixed_source = db.get(Source, fixed["source_id"])
        assert (fixed_source.assignment_mode, fixed_source.course_id) == ("explicit_course", space["id"])
        assert duplicate["deduplicated"] and duplicate["assignment_conflict"]
        assert db.get(Source, fixed["source_id"]).course_id == space["id"]
    engine.dispose()


def test_conversations_persist_and_delete(tmp_path, monkeypatch):
    client, _, engine = _scope(tmp_path, monkeypatch)
    created = client.post("/conversations", json={"title": "New chat"}).json()
    client.post(f"/conversations/{created['id']}/messages", json={"role": "user", "content": "Explain embeddings"})
    client.post(f"/conversations/{created['id']}/messages", json={"role": "assistant", "content": "A vector representation."})
    assert len(client.get(f"/conversations/{created['id']}").json()["messages"]) == 2
    assert any(row["id"] == created["id"] for row in client.get("/conversations").json())
    assert client.delete(f"/conversations/{created['id']}").status_code == 200
    assert client.get(f"/conversations/{created['id']}").status_code == 404
    engine.dispose()


def test_backend_rejects_study_pack_for_outline_node(tmp_path, monkeypatch):
    client, sessions, engine = _scope(tmp_path, monkeypatch)
    with sessions() as db:
        plan = StudyPlan(scope_key="all", revision="r1", syllabus="Outline")
        db.add(plan)
        db.flush()
        db.add(StudyPlanNode(plan_id=plan.id, node_id="T1", node_type="outline", title="Vectorization"))
        db.commit()
    response = client.post("/study-pack/export", json={"topic": "Vectorization", "topic_id": "T1"})
    assert response.status_code == 422
    engine.dispose()


def test_manual_resume_resets_retry_budget_but_keeps_checkpoint(tmp_path, monkeypatch):
    client, sessions, engine = _scope(tmp_path, monkeypatch)
    with sessions() as db:
        source=Source(name="Rate limited source", source_type="youtube", status="failed")
        db.add(source); db.flush()
        job=Job(source_id=source.id, task="process_source", status="failed", stage="index", retry_count=5, checkpoint="chunk:384/828", progress_total=828)
        db.add(job); db.commit(); job_id=job.id; source_id=source.id
    response=client.post(f"/jobs/{job_id}/retry")
    assert response.status_code == 200
    with sessions() as db:
        job=db.get(Job, job_id)
        assert (job.status, job.retry_count, job.checkpoint) == ("queued", 0, "chunk:384/828")
        assert db.get(Source, source_id).status == "queued"
    engine.dispose()


def test_old_embedding_429_job_shows_actionable_resume_message():
    from app.main import _job_error_message
    job=Job(source_id=1, status="failed", stage="organize", error="429 Too Many Requests at batchEmbedContents", error_message="Processing failed during organize.")
    message=_job_error_message(job)
    assert "rate limit or quota" in message
    assert "embedding" in message
    assert "resume" in message


def test_maintenance_backfill_job_has_single_worker_claim(tmp_path, monkeypatch):
    _, sessions, engine = _scope(tmp_path, monkeypatch)
    with sessions() as db:
        job = MaintenanceJob(task="semantic_group_backfill", status="queued", progress_total=0, items_json="[]")
        db.add(job)
        db.commit()
        job_id = job.id
    first = Worker()
    second = Worker()
    assert first._claim_maintenance_job() == job_id
    assert second._claim_maintenance_job() is None
    with sessions() as db:
        assert db.get(MaintenanceJob, job_id).status == "running"
    engine.dispose()


def test_organize_existing_assigns_unassigned_by_profile_and_preserves_manual_choice(tmp_path, monkeypatch):
    client, sessions, engine = _scope(tmp_path, monkeypatch)
    course = client.post("/courses", json={"name": "Natural Language Processing", "domain": "NLP"}).json()
    with sessions() as db:
        unassigned = Source(name="Lecture 1", source_type="transcript", status="done", assignment_mode="explicit_unassigned", assignment_status="unassigned", profile_json=json.dumps({"schema_version": 2, "course_title": "NLP Fundamentals", "primary_learning_goal": "Understand natural language processing", "domain": "NLP"}))
        fixed = Source(name="Deployment course", source_type="transcript", status="done", assignment_mode="explicit_course", assignment_status="assigned", course_id=course["id"], profile_json=json.dumps({"schema_version": 2, "course_title": "Deployment"}))
        db.add_all([unassigned, fixed]); db.commit()
        unassigned_id, fixed_id = unassigned.id, fixed.id

    class Classifier:
        def generate_json(self, *args, **kwargs):
            return {"course_id": course["id"], "new_course_name": "", "confidence": 0.92}

    monkeypatch.setattr(worker_module, "provider", lambda: Classifier())
    queued = client.post("/maintenance/organize-existing?include_unassigned=true").json()
    assert queued["progress_total"] == 3  # filing the unassigned source, then a placement check for each source
    Worker()._process_maintenance_job(queued["id"])
    with sessions() as db:
        assert db.get(Source, unassigned_id).course_id == course["id"]
        assert db.get(Source, fixed_id).course_id == course["id"]
        job = db.get(MaintenanceJob, queued["id"])
        assert (job.status, job.progress_current, job.progress_total) == ("done", 3, 3)
        assert json.loads(job.errors_json or "{}") == {}  # an off-topic AI answer to a placement check saves nothing and fails nothing
        assert db.query(SourceRelationship).count() == 0
    status = client.get("/maintenance/jobs").json()
    assert next(row for row in status if row["id"] == queued["id"])["status"] == "done"
    engine.dispose()


def test_maintenance_item_error_is_recorded_and_job_continues(tmp_path, monkeypatch):
    _, sessions, engine = _scope(tmp_path, monkeypatch)
    with sessions() as db:
        source = Source(name="Bad embedding", source_type="transcript", status="done")
        db.add(source); db.flush()
        document = Document(source_id=source.id, title="Lesson", status="done")
        db.add(document); db.flush()
        chunk = Chunk(document_id=document.id, chunk_index=0, chunk_kind="leaf", text="Text", embedding_status="done", embedding_json="{")
        db.add(chunk); db.flush()
        job = MaintenanceJob(task="organize_existing", status="running", progress_total=1, items_json=json.dumps([{"kind": "chunk", "id": chunk.id}]))
        db.add(job); db.commit(); job_id=job.id; chunk_id=chunk.id
    monkeypatch.setattr(worker_module, "provider", lambda: object())
    Worker()._process_maintenance_job(job_id)
    with sessions() as db:
        job=db.get(MaintenanceJob, job_id)
        assert job.status == "failed"
        assert job.progress_current == 1
        assert f"chunk:{chunk_id}" in json.loads(job.errors_json)
    engine.dispose()


def test_study_plan_reuses_persisted_source_revision(tmp_path, monkeypatch):
    client, sessions, engine = _scope(tmp_path, monkeypatch)
    from app.services import plans
    from tests.plan_jobs import run_plan_jobs
    calls = []
    monkeypatch.setattr(plans, "build_syllabus_structured", lambda db, ai, course_id, **options: (calls.append(course_id) or "| syllabus |", [{"node_id": "T1", "parent_node_id": None, "node_type": "study", "title": "Foundations", "metadata_json": "{}"}]))
    with sessions() as db:
        db.add(Source(name="Indexed source", source_type="transcript", status="done"))
        db.commit()
    client.get("/syllabus")
    run_plan_jobs(sessions)
    first = client.get("/syllabus").json()
    with sessions() as db:
        plan = db.query(StudyPlan).filter_by(scope_key="all").one()
        plan.relationships_json = "[]"
        db.commit()
    second = client.get("/syllabus").json()
    assert first["revision"] == second["revision"]
    assert second["nodes"][0]["id"] == "T1"
    assert second["source_relationships"] == {"sources": [], "cross_paths": [], "shared_foundations": []}
    assert len(calls) == 1
    engine.dispose()


def test_same_course_sources_form_one_path(tmp_path, monkeypatch):
    assert group_source_paths([1, 2], [{"source_a_id": 1, "source_b_id": 2, "relation_type": "SAME_COURSE"}]) == [[1, 2]]


def test_textbook_and_lecture_notes_can_be_complementary(tmp_path, monkeypatch):
    assert group_source_paths([1, 2], [{"source_a_id": 1, "source_b_id": 2, "relation_type": "COMPLEMENTARY_RESOURCE"}]) == [[1, 2]]


def test_overlapping_unrelated_courses_stay_in_separate_paths():
    rel = {"source_a_id": 1, "source_b_id": 2, "relation_type": "UNRELATED"}
    assert group_source_paths([1, 2], [rel]) == [[1], [2]]


def test_nlp_and_deployment_stay_separate_when_only_related():
    rel = {"source_a_id": 1, "source_b_id": 2, "relation_type": "RELATED_BUT_SEPARATE"}
    assert group_source_paths([1, 2], [rel]) == [[1], [2]]


def test_insufficient_identity_evidence_defaults_to_low_confidence_separate(tmp_path, monkeypatch):
    _, sessions, engine = _scope(tmp_path, monkeypatch)
    class FailingProvider:
        def generate_json(self, *args, **kwargs):
            raise RuntimeError("identity unavailable")
    with sessions() as db:
        a = Source(name="Unknown source one", source_type="text", status="done")
        b = Source(name="Unknown source two", source_type="text", status="done")
        db.add_all([a, b]); db.flush()
        rows = [{"id": a.id, "name": a.name, "uri": None, "user_space_id": None, "profile": {}, "concepts": ["embeddings"], "evidence": []},
                {"id": b.id, "name": b.name, "uri": None, "user_space_id": None, "profile": {}, "concepts": ["embeddings"], "evidence": []}]
        rels = _classify_relationships(db, FailingProvider(), rows)
        assert rels[0]["relation_type"] == "RELATED_BUT_SEPARATE"
        assert rels[0]["confidence"] == 0.2
        assert group_source_paths([a.id, b.id], rels) == [[a.id], [b.id]]
    engine.dispose()


def test_relationship_classification_defers_writes_until_plan_persistence(tmp_path, monkeypatch):
    _, sessions, engine = _scope(tmp_path, monkeypatch)
    class Provider:
        def generate_json(self, *args, **kwargs):
            return {"relationships": [{"source_a_id": 1, "source_b_id": 2, "type": "SAME_COURSE", "confidence": 0.9, "rationale": "Same course", "evidence": []}]}
    with sessions() as db:
        a=Source(name="Course lecture", source_type="text", status="done")
        b=Source(name="Course notes", source_type="text", status="done")
        db.add_all([a,b]); db.flush()
        rows=[{"id": a.id, "name": a.name, "uri": None, "user_space_id": None, "profile": {}, "concepts": [], "evidence": []},
              {"id": b.id, "name": b.name, "uri": None, "user_space_id": None, "profile": {}, "concepts": [], "evidence": []}]
        rels=_classify_relationships(db, Provider(), rows)
        assert rels[0]["relation_type"] == "SAME_COURSE"
        assert any(isinstance(item, SourceRelationship) for item in db.new)
    engine.dispose()


def test_shared_topic_word_does_not_merge_different_learning_purposes():
    rel = {"source_a_id": 1, "source_b_id": 2, "relation_type": "SAME_DOMAIN_DIFFERENT_COURSE"}
    assert group_source_paths([1, 2], [rel]) == [[1], [2]]
