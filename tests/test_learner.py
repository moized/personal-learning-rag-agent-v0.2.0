import json
from contextlib import contextmanager
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import (
    Base, Chunk, Concept, ConceptMastery, ConceptRelation, Course, Document, Evidence, Source,
)
from app.providers.local import LocalProvider
from app.services import learner
from app.services.learner import (
    add_misconception, due_reviews, effective_mastery, mastery_overview, now,
    open_misconceptions, recommend_next, record_result, sm2_update,
)
from app.services.quiz import generate_quiz_items, grade_answer, heuristic_score


class ScriptedProvider:
    """Returns canned JSON so grading and generation paths are deterministic."""

    def __init__(self, payload):
        self.payload = payload

    def generate_json(self, prompt, task="extraction", schema=None):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


@pytest.fixture()
def sessions(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'learner.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture()
def graph(sessions):
    """Course with Linear Algebra -> Eigenvalues -> PCA, each backed by one chunk."""
    with sessions() as db:
        course = Course(name="ML")
        db.add(course)
        db.flush()
        source = Source(name="Lecture", source_type="text", course_id=course.id, status="done")
        db.add(source)
        db.flush()
        document = Document(source_id=source.id, title="Lecture")
        db.add(document)
        db.flush()
        names = ["Linear Algebra", "Eigenvalues", "PCA"]
        concepts = [Concept(canonical_name=n) for n in names]
        db.add_all(concepts)
        db.flush()
        ids = {}
        for index, concept in enumerate(concepts):
            chunk = Chunk(document_id=document.id, chunk_index=index, chunk_kind="leaf", embedding_status="done",
                          text=f"{concept.canonical_name} explained with covariance matrices and projections.")
            db.add(chunk)
            db.flush()
            db.add(Evidence(concept_id=concept.id, chunk_id=chunk.id, score=1.0))
            ids[concept.canonical_name] = (concept.id, chunk.id)
        for a, b in (("Linear Algebra", "Eigenvalues"), ("Eigenvalues", "PCA")):
            db.add(ConceptRelation(from_concept_id=ids[a][0], to_concept_id=ids[b][0],
                                   relation_type="prerequisite", confidence=0.9,
                                   evidence_json=json.dumps({"chunk_ids": [ids[a][1]]})))
        db.commit()
        return {"course_id": course.id, **{n: v[0] for n, v in ids.items()}}


def test_sm2_resets_on_failure_and_grows_on_success():
    ease, interval, reps = sm2_update(2.5, 0.0, 0, 1.0)
    assert (interval, reps) == (1.0, 1)
    ease, interval, reps = sm2_update(ease, interval, reps, 1.0)
    assert (interval, reps) == (6.0, 2)
    ease, interval, reps = sm2_update(ease, interval, reps, 1.0)
    assert interval == pytest.approx(6.0 * ease)
    ease, interval, reps = sm2_update(ease, interval, reps, 0.0)
    assert (interval, reps) == (1.0, 0)
    assert ease >= 1.3


def test_record_result_updates_mastery_and_schedule(sessions, graph):
    with sessions() as db:
        row = record_result(db, graph["PCA"], 1.0, course_id=graph["course_id"])
        assert row.attempts == 1 and row.correct_count == 1
        assert row.mastery == pytest.approx(learner.LEARNING_RATE)
        assert row.next_due_at > now()
        before = row.mastery
        row = record_result(db, graph["PCA"], 1.0)
        assert row.mastery > before
        with pytest.raises(ValueError):
            record_result(db, 9999, 1.0)


def test_mastery_decays_without_review(sessions, graph):
    with sessions() as db:
        row = record_result(db, graph["PCA"], 1.0)
        fresh = effective_mastery(row, row.last_seen_at)
        later = effective_mastery(row, row.last_seen_at + timedelta(days=30))
        assert fresh == pytest.approx(row.mastery)
        assert later < fresh * 0.1


def test_due_reviews_only_lists_overdue_concepts(sessions, graph):
    with sessions() as db:
        record_result(db, graph["PCA"], 1.0)
        assert due_reviews(db, graph["course_id"]) == []
        row = db.query(ConceptMastery).one()
        row.next_due_at = now() - timedelta(days=2)
        db.flush()
        due = due_reviews(db, graph["course_id"])
        assert [d["concept"] for d in due] == ["PCA"]
        assert due[0]["overdue_days"] >= 2


def test_recommend_next_respects_prerequisites(sessions, graph):
    with sessions() as db:
        # Nothing learned yet: only the root of the prerequisite chain is ready.
        assert [r["concept"] for r in recommend_next(db, graph["course_id"])] == ["Linear Algebra"]
        # Mastering the root unlocks the next concept.
        for _ in range(6):
            record_result(db, graph["Linear Algebra"], 1.0)
        row = db.query(ConceptMastery).one()
        row.next_due_at = now() + timedelta(days=30)
        row.last_seen_at = now()
        db.flush()
        assert learner.effective_mastery(row) >= learner.MASTERED
        assert [r["concept"] for r in recommend_next(db, graph["course_id"])] == ["Eigenvalues"]


def test_recommend_next_prioritises_due_and_reports_blockers(sessions, graph):
    with sessions() as db:
        record_result(db, graph["PCA"], 0.0)  # failed PCA while its prerequisites are unseen
        row = db.query(ConceptMastery).one()
        row.next_due_at = now() - timedelta(days=1)
        db.flush()
        top = recommend_next(db, graph["course_id"])[0]
        assert top["concept"] == "PCA" and top["reason"] == "review due"
        assert top["blocking_prerequisites"] == ["Eigenvalues"]


def test_mastery_overview_statuses(sessions, graph):
    with sessions() as db:
        record_result(db, graph["Eigenvalues"], 1.0)
        overview = {c["concept"]: c for c in mastery_overview(db, graph["course_id"])}
        assert overview["PCA"]["status"] == "unseen"
        assert overview["Eigenvalues"]["status"] == "learning"
        assert overview["Eigenvalues"]["confidence"] == pytest.approx(1 / 3, abs=0.001)


def test_misconceptions_deduplicate_and_count(sessions, graph):
    with sessions() as db:
        first = add_misconception(db, graph["PCA"], "PCA  maximises variance in the labels")
        again = add_misconception(db, graph["PCA"], "pca maximises variance in the labels")
        assert first.id == again.id and again.occurrences == 2
        assert len(open_misconceptions(db, [graph["PCA"]])) == 1
        again.resolved = True
        db.flush()
        assert open_misconceptions(db, [graph["PCA"]]) == []


def test_quiz_generation_and_llm_grading_records_misconception(sessions, graph):
    with sessions() as db:
        concept = db.get(Concept, graph["PCA"])
        ai = ScriptedProvider({"questions": [{"question": "What does PCA maximise?", "reference_answer": "Variance"}]})
        item = generate_quiz_items(db, ai, concept, count=1, course_id=graph["course_id"])[0]
        assert item.question == "What does PCA maximise?"
        assert json.loads(item.evidence_chunk_ids_json)

        grader = ScriptedProvider({"score": 0.2, "feedback": "Not quite.", "misconception": "PCA minimises variance"})
        result = grade_answer(db, grader, item, "It minimises variance")
        assert result["correct"] is False
        assert result["misconception"] == "PCA minimises variance"
        assert result["reference_answer"] == "Variance"
        assert len(open_misconceptions(db, [graph["PCA"]])) == 1

        good = grade_answer(db, ScriptedProvider({"score": 1, "feedback": "Yes", "misconception": ""}), item, "Variance")
        assert good["correct"] is True and good["misconception"] is None


def test_quiz_offline_fallbacks(sessions, graph):
    with sessions() as db:
        concept = db.get(Concept, graph["PCA"])
        item = generate_quiz_items(db, LocalProvider(), concept, count=3, course_id=graph["course_id"])[0]
        assert "PCA" in item.question and item.reference_answer
        result = grade_answer(db, LocalProvider(), item, item.reference_answer)
        assert result["correct"] is True and "offline" in result["feedback"]
        wrong = grade_answer(db, ScriptedProvider(RuntimeError("down")), item, "banana")
        assert wrong["correct"] is False
        no_evidence = Concept(canonical_name="Orphan")
        db.add(no_evidence)
        db.flush()
        with pytest.raises(ValueError):
            generate_quiz_items(db, LocalProvider(), no_evidence)


def test_heuristic_score_bounds():
    assert heuristic_score("", "gradient descent minimises loss") == 0.0
    assert heuristic_score("gradient descent minimises loss", "gradient descent minimises loss") == 1.0
    assert heuristic_score("anything", "") == 0.0


def test_learner_endpoints(sessions, graph, monkeypatch):
    from app.routers import learner as router_module
    from app import main

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

    monkeypatch.setattr(router_module, "session_scope", scope)
    monkeypatch.setattr(router_module, "provider", lambda: LocalProvider())
    client = TestClient(main.app)

    generated = client.post("/quiz/generate", json={"concept_name": "pca", "course_id": graph["course_id"]})
    assert generated.status_code == 200
    item = generated.json()["items"][0]
    assert "reference_answer" not in item

    answered = client.post(f"/quiz/{item['id']}/answer", json={"answer": "covariance matrices and projections"})
    assert answered.status_code == 200 and "score" in answered.json()

    assert client.post("/quiz/generate", json={"concept_name": "nope"}).status_code == 404
    assert client.post("/quiz/999/answer", json={"answer": "x"}).status_code == 404

    rated = client.post("/learner/self-rating", json={"concept_id": graph["Eigenvalues"], "rating": 5})
    assert rated.status_code == 200
    assert client.post("/learner/self-rating", json={"concept_id": 9999, "rating": 3}).status_code == 404
    assert client.post("/learner/self-rating", json={"concept_id": 1, "rating": 9}).status_code == 422

    mastery = client.get("/learner/mastery", params={"course_id": graph["course_id"]}).json()["concepts"]
    assert {c["concept"] for c in mastery} == {"Linear Algebra", "Eigenvalues", "PCA"}
    assert client.get("/learner/due").json() == {"due": []}
    assert "next" in client.get("/learner/next", params={"course_id": graph["course_id"]}).json()

    with sessions() as db:
        miscon = add_misconception(db, graph["PCA"], "wrong belief")
        db.commit()
        miscon_id = miscon.id
    listed = client.get("/learner/misconceptions").json()["misconceptions"]
    assert listed[0]["concept"] == "PCA"
    assert client.post(f"/learner/misconceptions/{miscon_id}/resolve").json()["resolved"] is True
    assert client.get("/learner/misconceptions").json()["misconceptions"] == []
    assert client.post("/learner/misconceptions/999/resolve").status_code == 404
