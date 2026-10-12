"""Removing material and study spaces completely, including what LearnOS made from them and the deleted text on disk."""
import json

from sqlalchemy import create_engine, event, select, text

from app.models import (
    AgentTrace, Chunk, Concept, ConceptMastery, Course, Document, Evidence, Group, LearnerEvent, MaintenanceJob, Misconception,
    QuizItem, Source, SourceSpace, StudyPlan, StudyPlanNode, StudyProgress, TopicDetail,
)
from app.services.cleanup import COMPACT_TASK, forget_empty_scopes
from tests.test_plans import _add_source, _course, env  # noqa: F401


def _space(db, name, created_by=None):
    course = _course(db, name)
    course.created_by = created_by
    db.flush()
    return course


def _chunks_of(db, source_id):
    return db.execute(select(Chunk).join(Document).where(Document.source_id == source_id)).scalars().all()


def test_removing_material_removes_everything_made_from_it(env):
    with env.sessions() as db:
        course = _space(db, "Deep Learning", "you")
        gone = _add_source(db, course.id, "Gone", ["Backprop"])
        kept = _add_source(db, course.id, "Kept", ["Momentum"])
        gone_chunk, kept_chunk = _chunks_of(db, gone.id)[0], _chunks_of(db, kept.id)[0]
        group = Group(name="Only gone")
        db.add(group)
        db.flush()
        gone_chunk.group_id = group.id
        concept = db.execute(select(Concept).where(Concept.canonical_name == "Backprop")).scalar_one()
        quiz = QuizItem(concept_id=concept.id, course_id=course.id, question="What does backprop compute?", reference_answer="Gradients")
        db.add(quiz)
        db.flush()
        db.add_all([ConceptMastery(concept_id=concept.id), Misconception(concept_id=concept.id, description="It is magic"),
                    LearnerEvent(concept_id=concept.id, quiz_item_id=quiz.id, course_id=course.id, event_type="answer"),
                    TopicDetail(scope_key=f"course:{course.id}", node_id="t-1", budget_tokens=8000, title="Backprop", plan_revision="r",
                                status="unit", payload_json=json.dumps({"study_text": "Backprop explained."})),
                    TopicDetail(scope_key="all", node_id="t-1", budget_tokens=8000, title="Backprop", plan_revision="r",
                                status="unit", payload_json=json.dumps({"study_text": "Backprop explained."})),
                    AgentTrace(run_id="a", event_type="study", result_json=json.dumps({"retrieved_chunk_ids": [gone_chunk.id]})),
                    AgentTrace(run_id="b", event_type="study", result_json=json.dumps({"retrieved_chunk_ids": [kept_chunk.id]}))])
        db.commit()
        gone_id, course_id = gone.id, course.id
    assert env.client.delete(f"/sources/{gone_id}").status_code == 200
    with env.sessions() as db:
        assert db.query(Source).count() == 1 and db.get(Course, course_id) is not None  # the learner's own space stays
        assert db.query(Concept).filter_by(canonical_name="Backprop").count() == 0
        assert db.query(QuizItem).count() == db.query(Misconception).count() == db.query(ConceptMastery).count() == db.query(LearnerEvent).count() == 0
        assert db.query(TopicDetail).count() == 0  # prepared sessions that may quote it
        assert [t.run_id for t in db.query(AgentTrace).all()] == ["b"]  # only logs that used its passages
        assert db.query(Group).filter_by(name="Only gone").count() == 0
        assert db.query(MaintenanceJob).filter_by(task=COMPACT_TASK).count() == 1  # the file is compacted afterwards


def test_a_space_learnos_made_is_removed_with_its_last_material(env):
    with env.sessions() as db:
        made = _space(db, "Made by LearnOS", "learnos")
        own = _space(db, "Made by me", "you")
        a = _add_source(db, made.id, "A", ["Topic A"])
        b = _add_source(db, own.id, "B", ["Topic B"])
        db.commit()
        ids = (a.id, b.id, made.id, own.id)
    removed = env.client.delete(f"/sources/{ids[0]}").json()
    assert removed["removed_spaces"] == [ids[2]]
    env.client.delete(f"/sources/{ids[1]}")
    with env.sessions() as db:
        assert db.get(Course, ids[2]) is None and db.get(Course, ids[3]) is not None


def test_deleting_a_space_keeps_its_material_in_all_sources(env):
    with env.sessions() as db:
        course = _space(db, "Statistics", "you")
        other = _space(db, "Other", "you")
        source = _add_source(db, course.id, "Lecture", ["Tests"])
        guest = _add_source(db, other.id, "Guest", ["More"])
        db.add(SourceSpace(source_id=guest.id, course_id=course.id))
        plan = StudyPlan(scope_key=f"course:{course.id}", revision="r", syllabus="")
        db.add(plan)
        db.flush()
        db.add_all([StudyPlanNode(plan_id=plan.id, node_id="t-1", node_type="study", path_id="p", title="Tests"),
                    StudyProgress(scope_key=f"course:{course.id}", topic_key="tests", topic_label="Tests", node_id="t-1"),
                    TopicDetail(scope_key=f"course:{course.id}", node_id="t-1", budget_tokens=8000, title="Tests", plan_revision="r",
                                status="unit", payload_json="{}")])
        db.commit()
        course_id, source_id = course.id, source.id
    listed = {c["name"]: c for c in env.client.get("/courses").json()}
    assert listed["Statistics"]["source_count"] == 2 and listed["Statistics"]["created_by"] == "you"
    assert env.client.delete(f"/courses/{course_id}").json()["removed_sources"] == 0
    with env.sessions() as db:
        assert db.get(Course, course_id) is None
        source = db.get(Source, source_id)
        assert source.course_id is None and "was deleted" in source.assignment_reason
        assert db.query(SourceSpace).count() == 0 and db.query(StudyPlan).count() == 0 and db.query(StudyPlanNode).count() == 0
        assert db.query(StudyProgress).count() == 0 and db.query(TopicDetail).count() == 0
    assert env.client.delete(f"/courses/{course_id}").status_code == 404


def test_deleting_a_space_with_its_material_removes_the_material(env):
    with env.sessions() as db:
        course = _space(db, "Statistics", "you")
        _add_source(db, course.id, "Lecture", ["Tests"])
        db.commit()
        course_id = course.id
    assert env.client.delete(f"/courses/{course_id}", params={"with_material": "true"}).json()["removed_sources"] == 1
    with env.sessions() as db:
        assert db.query(Source).count() == 0 and db.query(Chunk).count() == 0 and db.query(Evidence).count() == 0


def test_study_data_left_for_spaces_without_material_is_cleared(env):
    with env.sessions() as db:  # what an older version left behind after the material was removed
        db.add_all([TopicDetail(scope_key="all", node_id="t-1", budget_tokens=8000, title="Old", plan_revision="r", status="unit",
                                payload_json=json.dumps({"study_text": "Excerpts of removed material."})),
                    StudyProgress(scope_key="all", topic_key="old", topic_label="Old", node_id="t-1"),
                    StudyProgress(scope_key="course:99", topic_key="old", topic_label="Old", node_id="t-2")])
        course = _space(db, "Live", "you")
        _add_source(db, course.id, "Lecture", ["Tests"])
        db.add(StudyProgress(scope_key=f"course:{course.id}", topic_key="tests", topic_label="Tests", node_id="t-3"))
        db.commit()
    with env.sessions() as db:
        db.query(Source).delete()  # no material in All sources at all...
        db.commit()
    with env.sessions() as db:
        forget_empty_scopes(db)
        db.commit()
    with env.sessions() as db:
        assert db.query(TopicDetail).count() == 0 and db.query(StudyProgress).count() == 0


def test_deleted_text_is_not_left_in_the_database_file(tmp_path):
    from app.db import compact_storage
    path = tmp_path / "erase.db"
    engine = create_engine(f"sqlite:///{path}")

    @event.listens_for(engine, "connect")
    def _secure(dbapi_connection, _record):
        dbapi_connection.execute("PRAGMA secure_delete=ON")

    marker = "UNMISTAKABLE-MARKER-" * 50
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        conn.execute(text("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)"))
        for _ in range(200):
            conn.execute(text("INSERT INTO notes (body) VALUES (:body)"), {"body": marker})
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM notes"))
    compact_storage(engine)
    engine.dispose()
    leftovers = b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.name.startswith("erase.db"))
    assert b"UNMISTAKABLE-MARKER" not in leftovers
