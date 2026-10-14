"""End to end in manual mode: every model decision is answered by "pasting" an external AI reply."""
import json
import re
import pytest
from fastapi.testclient import TestClient
from app.config import settings
from app.main import app
from app.models import Chunk, Decision, Group, Job, Source
from app.providers.base import DecisionPending
from app.services.decisions import pause_job
from app.services.queue import enqueue
from app.services.tutor import _parts, parse_start_reply
from app.services.worker import Worker
from tests.helpers import BM25, DENSE, RRF, add_course


def excerpts(prompt):
    return [int(n) for n in re.findall(r"^\[(\d+)\]", prompt, flags=re.MULTILINE)]


def run_job(db, job_id):
    """Run the worker on one job; on a pending decision, pause it like the worker loop does."""
    try:
        Worker().process_job(job_id)
    except DecisionPending as pending:
        pause_job(db, job_id, pending)
        db.commit()
        return db.get(Decision, db.get(Job, job_id).waiting_decision_id)
    db.expire_all()
    return None


def add_transcript(db, tmp_path, course, name, paragraphs):
    path = tmp_path / f"{name}.txt"
    path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    src = Source(name=name, source_type="transcript", local_path=str(path), course_id=course.id, status="queued")
    db.add(src); db.flush()
    job = enqueue(db, src.id)
    db.commit()
    return src, job


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 300)
    monkeypatch.setattr(settings, "chunk_overlap", 40)


def test_manual_mode_from_source_to_one_topic_at_a_time(db, tmp_path, small_chunks):
    client = TestClient(app)
    assert client.get("/health").json()["ai_mode"] == "manual"
    course = add_course(db)
    src1, job1 = add_transcript(db, tmp_path, course, "Lecture 1", [BM25, DENSE])
    src2, job2 = add_transcript(db, tmp_path, course, "Lecture 2", [RRF, BM25])

    # 1. The first source splits into pieces, then pauses for an organize decision.
    decision = run_job(db, job1.id)
    assert decision is not None and db.get(Job, job1.id).status == "waiting"
    pending = client.get("/decisions").json()
    assert [d["id"] for d in pending] == [decision.id]
    assert "Organize 'Lecture 1'" in pending[0]["label"]
    assert excerpts(pending[0]["prompt"]) == [1, 2]
    assert "Reply with only a JSON object" in pending[0]["prompt"]

    # A malformed paste is rejected without losing the decision.
    assert client.post(f"/decisions/{decision.id}/answer", json={"answer": "group 1 please"}).status_code == 400

    reply = {
        "assignments": [{"excerpt": 1, "group": "Lexical retrieval", "concepts": ["BM25"]},
                        {"excerpt": 2, "group": "Dense retrieval", "concepts": ["Embedding"]}],
        "new_groups": [{"name": "Lexical retrieval", "description": "Term matching."},
                       {"name": "Dense retrieval", "description": "Vector similarity."}],
        "new_concepts": [{"name": "BM25", "description": "Ranking function."},
                         {"name": "Embedding", "description": "Vector for text."}],
    }
    result = client.post(f"/decisions/{decision.id}/answer", json={"answer": "```json\n" + json.dumps(reply) + "\n```"}).json()
    assert result["requeued_jobs"] == [job1.id]
    assert run_job(db, job1.id) is None
    assert db.get(Job, job1.id).status == "done"

    # 2. The second source repeats the BM25 paragraph: it is merged, so only RRF needs a decision.
    decision = run_job(db, job2.id)
    prompt = client.get(f"/decisions/{decision.id}").json()["prompt"]
    assert excerpts(prompt) == [1]
    assert "- Lexical retrieval: Term matching." in prompt
    reply = {"assignments": [{"excerpt": 1, "group": "Rank fusion", "concepts": ["Reciprocal rank fusion"]}],
             "new_concepts": [{"name": "Reciprocal rank fusion", "description": "Fuses rankings.", "aliases": ["RRF"]}]}
    client.post(f"/decisions/{decision.id}/answer", json={"answer": json.dumps(reply)})
    assert run_job(db, job2.id) is None
    recap = db.query(Chunk).filter(Chunk.duplicate_of_id.is_not(None)).one()
    lexical = db.query(Group).filter_by(name="Lexical retrieval").one()
    assert recap.group_id == lexical.id and recap.classification_status == "merged"
    assert client.get(f"/courses/{course.id}/stats").json()["merged_duplicates"] == 1

    # 3. The study outline is one more decision; answering it finishes the syllabus.
    assert client.get("/syllabus", params={"course_id": course.id}).json()["status"] == "missing"
    response = client.post("/syllabus", json={"course_id": course.id})
    assert response.status_code == 202
    outline_decision = response.json()["decision"]
    assert "## Lexical retrieval" in outline_decision["prompt"]
    outline = {"topics": [
        {"title": "Lexical retrieval", "purpose": "Rank by terms.", "groups": ["Lexical retrieval"], "concepts": ["BM25"]},
        {"title": "Dense retrieval", "purpose": "Rank by meaning.", "prerequisites": [1], "groups": ["Dense retrieval"], "concepts": ["Embedding"]},
        {"title": "Hybrid fusion", "purpose": "Combine both.", "prerequisites": [1, 2], "groups": ["Rank fusion"], "concepts": ["RRF"]},
    ]}
    done = client.post(f"/decisions/{outline_decision['id']}/answer", json={"answer": json.dumps(outline)}).json()
    topics = done["result"]["topics"]
    assert [t["title"] for t in topics] == ["Lexical retrieval", "Dense retrieval", "Hybrid fusion"]
    assert [t["excerpts"] for t in topics] == [1, 1, 1]
    assert all(t["status"] == "pending" for t in topics)

    # 4. Kickoff: the external AI gets the outline and is told to ask where to start.
    kickoff = client.get("/tutor/kickoff", params={"course_id": course.id}).json()["text"]
    assert "2. Dense retrieval — Rank by meaning. (builds on 1)" in kickoff
    assert "ask me where I want to start" in kickoff and "START: 3; SKIP: 5, 7" in kickoff

    # 5. The user pastes the tutor's decision back: start at 2 and skip nothing.
    first = client.post("/tutor/start", json={"course_id": course.id, "reply": "Great choice.\nSTART: 2; SKIP: none"}).json()
    assert first["number"] == 2 and first["part"] == 1 and first["parts"] == 1
    assert first["text"].startswith("TOPIC 2 of 3 — Dense retrieval")
    assert "Builds on: topic 1 (Lexical retrieval)" in first["text"]
    # Verbatim material without the overlap copied from the BM25 chunk, which belongs to topic 1.
    assert DENSE in first["text"] and "technical terms" not in first["text"]

    assert client.post("/tutor/start", json={"course_id": course.id, "start": 9}).status_code == 400
    assert client.post("/tutor/start", json={"course_id": course.id, "reply": "Let's begin!"}).status_code == 400

    second = client.post("/tutor/next", json={"course_id": course.id}).json()
    assert second["number"] == 3 and RRF in second["text"] and second["remaining_topics"] == 0
    assert client.post("/tutor/next", json={"course_id": course.id}).json()["done"] is True

    statuses = [t["status"] for t in client.get("/syllabus", params={"course_id": course.id}).json()["topics"]]
    assert statuses == ["skipped", "done", "done"]

    # 6. Un-skipping topic 1 brings it back; re-sending a part does not change progress.
    topic1 = topics[0]["id"]
    assert client.post(f"/syllabus/topics/{topic1}/status", json={"status": "pending"}).json()["status"] == "pending"
    resend = client.get(f"/tutor/topics/{topic1}/parts/1").json()
    assert BM25 in resend["text"] and "same text also in: Lecture 2" in resend["text"]
    nxt = client.post("/tutor/next", json={"course_id": course.id}).json()
    assert nxt["number"] == 1


def test_regenerating_the_outline_keeps_topic_status(db, tmp_path, small_chunks, monkeypatch, scripted):
    from app.services import curriculum
    course = add_course(db)
    src, job = add_transcript(db, tmp_path, course, "Lecture 1", [BM25, DENSE])
    ai = scripted(organize=lambda p: {"assignments": [
        {"excerpt": n, "group": "Retrieval", "concepts": []} for n in excerpts(p)]})
    monkeypatch.setattr("app.services.worker.provider", lambda: ai)
    Worker().process_job(job.id)

    ai.responses["syllabus"] = lambda p: {"topics": [{"title": "Retrieval", "purpose": "x", "groups": ["Retrieval"], "concepts": []}]}
    syllabus = curriculum.generate_syllabus(db, ai, course.id)
    topic = curriculum.syllabus_topics(db, syllabus)[0]
    topic.status = "done"
    db.flush()
    ai.responses["syllabus"] = lambda p: {"topics": [
        {"title": "Basics", "purpose": "y", "groups": [], "concepts": []},
        {"title": "retrieval", "purpose": "x", "groups": ["Retrieval"], "concepts": []}]}
    syllabus = curriculum.generate_syllabus(db, ai, course.id)
    assert [(t.title, t.status) for t in curriculum.syllabus_topics(db, syllabus)] == [("Basics", "pending"), ("retrieval", "done")]


def test_parse_start_reply_and_part_splitting(monkeypatch):
    assert parse_start_reply("START: 3; SKIP: 5, 7-9") == (3, [5, 7, 8, 9])
    assert parse_start_reply("start 1\nskip: none") == (1, [])
    assert parse_start_reply("no idea") == (None, [])
    monkeypatch.setattr(settings, "study_part_chars", 25)
    assert _parts(["a" * 20, "b" * 20, "c" * 5]) == [["a" * 20], ["b" * 20], ["c" * 5]]
    assert _parts([]) == [[]]
