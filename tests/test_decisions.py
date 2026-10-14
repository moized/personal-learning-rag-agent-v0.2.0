import pytest
from app.models import Decision, Job, Source
from app.providers.base import DecisionPending
from app.providers.local import LocalProvider
from app.providers.manual import ManualProvider, schema_example
from app.services.decisions import pause_job, record_decision, requeue_waiting_jobs, submit_answer

SCHEMA = {
    "type": "object",
    "properties": {"group_id": {"type": "integer"}, "names": {"type": "array", "items": {"type": "string"}}},
    "required": ["group_id"],
}


def manual(db):
    from app.db import SessionLocal
    return ManualProvider(LocalProvider(32).embed, SessionLocal)


def test_schema_example_shows_the_expected_shape():
    assert schema_example(SCHEMA) == {"group_id": 0, "names": ["..."]}


def test_manual_provider_asks_then_returns_the_pasted_answer(db):
    ai = manual(db)
    with pytest.raises(DecisionPending) as info:
        ai.generate_json("Pick a group.", task="routing", schema=SCHEMA, label="Pick")
    pending = info.value
    assert "Pick a group." in pending.prompt
    assert '"group_id": 0' in pending.prompt

    decision = record_decision(db, pending)
    db.commit()
    assert decision.status == "pending"

    # Chat AIs often wrap JSON in prose and code fences.
    submit_answer(db, decision.id, 'Sure! Here it is:\n```json\n{"group_id": 7, "names": ["a"]}\n```')
    db.commit()
    assert ai.generate_json("Pick a group.", task="routing", schema=SCHEMA) == {"group_id": 7, "names": ["a"]}


def test_answer_must_be_json_with_required_fields(db):
    with pytest.raises(DecisionPending) as info:
        manual(db).generate_json("Pick a group.", task="routing", schema=SCHEMA)
    decision = record_decision(db, info.value)
    with pytest.raises(ValueError, match="group_id"):
        submit_answer(db, decision.id, '{"names": []}')
    with pytest.raises(ValueError):
        submit_answer(db, decision.id, "I think group 7 is best.")
    assert decision.status == "pending"


def test_text_decisions_store_the_raw_reply(db):
    ai = manual(db)
    with pytest.raises(DecisionPending) as info:
        ai.generate_text("Summarize.", task="study")
    decision = record_decision(db, info.value)
    submit_answer(db, decision.id, "  A summary.  ")
    db.commit()
    assert ai.generate_text("Summarize.", task="study") == "A summary."


def test_waiting_job_resumes_only_when_its_decision_is_answered(db):
    src = Source(name="s", source_type="file", status="processing")
    db.add(src); db.flush()
    job = Job(source_id=src.id, task="process_source", status="running", stage="organize")
    db.add(job); db.flush()

    first = pause_job(db, job.id, DecisionPending("organize:a", "organize", "json", "p1", SCHEMA, "batch 1"))
    assert job.status == "waiting" and src.status == "waiting"
    second = pause_job(db, job.id, DecisionPending("organize:b", "organize", "json", "p2", SCHEMA, "batch 1 again"))
    assert first.status == "superseded"
    assert job.waiting_decision_id == second.id

    assert requeue_waiting_jobs(db, second.id) == []
    submit_answer(db, second.id, '{"group_id": 1}')
    assert requeue_waiting_jobs(db, second.id) == [job.id]
    assert job.status == "queued" and src.status == "queued"
    assert db.get(Decision, second.id).job_id == job.id


def test_a_new_outline_request_supersedes_the_pending_one(db):
    old = record_decision(db, DecisionPending("syllabus:a", "syllabus", "json", "old", SCHEMA, "Outline"),
                          action="syllabus", action_args={"course_id": 1})
    other_course = record_decision(db, DecisionPending("syllabus:c", "syllabus", "json", "x", SCHEMA, "Outline"),
                                   action="syllabus", action_args={"course_id": 2})
    new = record_decision(db, DecisionPending("syllabus:b", "syllabus", "json", "new", SCHEMA, "Outline"),
                          action="syllabus", action_args={"course_id": 1})
    assert (old.status, other_course.status, new.status) == ("superseded", "pending", "pending")


def test_upgrade_marks_existing_chunks_as_canonical(db):
    from sqlalchemy import text
    from app.db import engine, init_db, Base
    Base.metadata.drop_all(engine)
    with engine.begin() as conn:
        # The v0.2 chunks table, before duplicate tracking existed.
        conn.execute(text("CREATE TABLE chunks (id INTEGER PRIMARY KEY, document_id INTEGER, chunk_index INTEGER, "
                          "text TEXT, embedding_status VARCHAR(30), classification_status VARCHAR(30), cluster_status VARCHAR(30))"))
        conn.execute(text("INSERT INTO chunks (id, document_id, chunk_index, text) VALUES (1, 1, 0, 'old text')"))
    init_db()
    with engine.begin() as conn:
        assert conn.execute(text("SELECT dedupe_status, duplicate_of_id FROM chunks")).one() == ("done", None)
