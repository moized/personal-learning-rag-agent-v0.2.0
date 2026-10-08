import json
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.models import (
    Base, Chunk, Concept, ConceptRelation, Document, Evidence, Group, Job, Source,
    SourceRelationship, StudyPlan,
)
from app.providers.base import ProviderAuthenticationError, ProviderRateLimitError
from app.services import worker as worker_module
from app.services.grouping import extract_group_concepts
from app.services.knowledge_graph import extract_concept_relations
from app.services.learning_paths import _classify_relationships, build_syllabus_structured
from app.services.queue import enqueue
from app.services.reliability import is_retryable_error, profile_needs_refresh
from app.services.worker import Worker

GOOD_PROFILE = {
    "author": "", "institution": "", "provider": "", "domain": "ml", "subdomain": "", "estimated_level": "",
    "course_title": "Intro to ML", "primary_learning_goal": "learn ml", "learning_objectives": ["x"],
    "expected_prerequisites": [], "learner_capabilities": [], "downstream_applications": [],
    "source_role": "lecture material", "course_identity_confidence": 0.8, "learning_intent_confidence": 0.8,
    "inferred_fields": [], "unknown_fields": [], "topics": ["x"], "description": "", "suggested_group_names": [],
}


class Scripted:
    """Provider whose JSON answer (or error) is chosen per task."""

    def __init__(self, answers=None, errors=None):
        self.answers, self.errors, self.calls = answers or {}, errors or {}, []

    def generate_json(self, prompt, task="extraction", schema=None):
        self.calls.append(task)
        if task in self.errors:
            raise self.errors[task]
        return self.answers.get(task, {})

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def generate_text(self, prompt, task="study"):
        return ""


@pytest.fixture()
def sessions(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'rel.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    engine.dispose()


def _source_with_chunk(db, name="Lecture", text="Gradient descent lowers the loss."):
    source = Source(name=name, source_type="transcript", status="processing")
    db.add(source)
    db.flush()
    doc = Document(source_id=source.id, title=name)
    db.add(doc)
    db.flush()
    chunk = Chunk(document_id=doc.id, chunk_index=0, chunk_kind="leaf", text=text, embedding_status="done")
    db.add(chunk)
    db.flush()
    return source, chunk


def test_retryable_error_classification():
    assert is_retryable_error(ProviderRateLimitError("slow down"))
    assert is_retryable_error(ProviderAuthenticationError("bad key"))
    assert is_retryable_error(RuntimeError("All Gemini JSON models failed: m: temporary Gemini error 503"))
    assert is_retryable_error(RuntimeError("HTTP 429 Too Many Requests"))
    assert is_retryable_error(RuntimeError("read timed out"))
    assert not is_retryable_error(ValueError("invalid JSON from model"))
    assert not is_retryable_error(KeyError("course_title"))


def test_profile_needs_refresh_detects_degraded_and_legacy_profiles():
    assert profile_needs_refresh({})
    assert profile_needs_refresh({"schema_version": 1, "course_title": "x"})
    assert profile_needs_refresh({**GOOD_PROFILE, "schema_version": 2, "degraded": True})
    # What older versions saved after a failed AI call: current-looking but empty.
    legacy = {"schema_version": 2, "course_title": "", "primary_learning_goal": "", "course_identity_confidence": 0.0}
    assert profile_needs_refresh(legacy)
    assert not profile_needs_refresh({**GOOD_PROFILE, "schema_version": 2})


def test_profile_outage_raises_instead_of_saving_an_empty_profile(sessions):
    with sessions() as db:
        source, _ = _source_with_chunk(db)
        with pytest.raises(RuntimeError):
            Worker()._ensure_source_profile(db, source, Scripted(errors={"classification": RuntimeError("HTTP 429 rate limit")}))
        assert source.profile_json is None


def test_unexpected_profile_error_saves_a_degraded_profile_that_is_retried_later(sessions):
    with sessions() as db:
        source, _ = _source_with_chunk(db)
        Worker()._ensure_source_profile(db, source, Scripted(errors={"classification": ValueError("model returned nonsense")}))
        degraded = json.loads(source.profile_json)
        assert degraded["degraded"] is True and profile_needs_refresh(degraded)

        healthy = Scripted(answers={"classification": GOOD_PROFILE})
        Worker()._ensure_source_profile(db, source, healthy)
        repaired = json.loads(source.profile_json)
        assert healthy.calls == ["classification"]
        assert repaired["course_title"] == "Intro to ML" and not repaired.get("degraded")

        Worker()._ensure_source_profile(db, source, healthy)
        assert healthy.calls == ["classification"]  # a good profile is not regenerated


def test_job_is_retried_when_gemini_is_rate_limited_during_profiling(tmp_path, sessions, monkeypatch):
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

    path = tmp_path / "notes.txt"
    path.write_text("Gradient descent lowers the loss.\n\nThe learning rate sets the step size.", encoding="utf-8")
    with sessions() as db:
        source = Source(name="notes.txt", source_type="file", local_path=str(path), status="queued", assignment_mode="auto")
        db.add(source)
        db.flush()
        job = enqueue(db, source.id)
        db.commit()
        source_id, job_id = source.id, job.id

    monkeypatch.setattr(worker_module, "session_scope", scope)
    monkeypatch.setattr(worker_module, "provider", lambda: Scripted(errors={"classification": RuntimeError("HTTP 429 rate limit")}))
    worker = Worker()
    with pytest.raises(RuntimeError) as caught:
        worker.process_job(job_id)
    assert worker._record_failure(job_id, caught.value) == 1

    with sessions() as db:
        job = db.get(Job, job_id)
        assert job.status == "retrying"
        assert "rate limit" in job.error_message.lower()
        assert db.get(Source, source_id).profile_json is None


def test_failed_source_comparison_is_not_saved_and_is_retried(sessions):
    with sessions() as db:
        a, _ = _source_with_chunk(db, "A")
        b, _ = _source_with_chunk(db, "B")
        rows = [{"id": s.id, "name": s.name, "uri": None, "user_space_id": None, "profile": {}, "concepts": [], "evidence": []} for s in (a, b)]

        warnings: list[str] = []
        rels = _classify_relationships(db, Scripted(errors={"source_relationship": RuntimeError("HTTP 503")}), rows, warnings)
        assert rels[0]["relation_type"] == "RELATED_BUT_SEPARATE" and rels[0]["provisional"] is True
        assert warnings and "unavailable" in warnings[0]
        assert not [item for item in db.new if isinstance(item, SourceRelationship)]
        db.flush()

        answer = {"relationships": [{"source_a_id": a.id, "source_b_id": b.id, "type": "SAME_COURSE", "confidence": 0.9, "rationale": "same", "evidence": []}]}
        warnings = []
        rels = _classify_relationships(db, Scripted(answers={"source_relationship": answer}), rows, warnings)
        assert rels[0]["relation_type"] == "SAME_COURSE" and not rels[0].get("provisional")
        assert warnings == []
        db.flush()
        assert db.execute(select(SourceRelationship)).scalar_one().relation_type == "SAME_COURSE"


def test_pair_skipped_by_the_model_is_provisional_not_saved(sessions):
    with sessions() as db:
        a, _ = _source_with_chunk(db, "A")
        b, _ = _source_with_chunk(db, "B")
        rows = [{"id": s.id, "name": s.name, "uri": None, "user_space_id": None, "profile": {}, "concepts": [], "evidence": []} for s in (a, b)]
        warnings: list[str] = []
        rels = _classify_relationships(db, Scripted(answers={"source_relationship": {"relationships": []}}), rows, warnings)
        assert rels[0]["provisional"] is True
        assert any("could not be compared" in w for w in warnings)
        assert not [item for item in db.new if isinstance(item, SourceRelationship)]


def test_default_guess_saved_by_older_versions_is_asked_again(sessions):
    with sessions() as db:
        a, _ = _source_with_chunk(db, "A")
        b, _ = _source_with_chunk(db, "B")
        rows = [{"id": s.id, "name": s.name, "uri": None, "user_space_id": None, "profile": {}, "concepts": [], "evidence": []} for s in (a, b)]
        from app.services.learning_paths import _digest
        legacy = SourceRelationship(
            source_a_id=a.id, source_b_id=b.id, source_a_profile_hash=_digest({}), source_b_profile_hash=_digest({}),
            relation_type="RELATED_BUT_SEPARATE", confidence=0.2,
            rationale="Not enough evidence to confirm a shared learning path.", evidence_json="[]")
        db.add(legacy)
        db.flush()
        answer = {"relationships": [{"source_a_id": a.id, "source_b_id": b.id, "type": "SAME_COURSE", "confidence": 0.9, "rationale": "same", "evidence": []}]}
        provider = Scripted(answers={"source_relationship": answer})
        rels = _classify_relationships(db, provider, rows, [])
        assert provider.calls == ["source_relationship"]
        assert rels[0]["relation_type"] == "SAME_COURSE"


def test_concept_extraction_outage_is_not_swallowed(sessions):
    with sessions() as db:
        _, chunk = _source_with_chunk(db)
        group = Group(name="ML")
        db.add(group)
        db.flush()
        chunk.group_id = group.id
        db.flush()
        with pytest.raises(RuntimeError):
            extract_group_concepts(db, Scripted(errors={"extraction": RuntimeError("HTTP 429")}), group)
        assert extract_group_concepts(db, Scripted(errors={"extraction": ValueError("bad json")}), group) == 0


def test_relation_extraction_outage_is_not_swallowed(sessions):
    with sessions() as db:
        one, two = Concept(canonical_name="One"), Concept(canonical_name="Two")
        db.add_all([one, two])
        db.flush()
        ids = {one.id, two.id}
        with pytest.raises(RuntimeError):
            extract_concept_relations(db, Scripted(errors={"relationship_extraction": RuntimeError("HTTP 503")}), ids, "test")
        assert extract_concept_relations(db, Scripted(errors={"relationship_extraction": ValueError("bad")}), ids, "test") == 0


def _plan_provider(fail_syllabus: bool):
    node = {"prerequisites": "", "prerequisite_confidence": 0, "why_included": "", "enables": "", "source_ids": [],
            "source_roles": [], "topic_merge_confidence": 0.5, "core_or_supplementary": "core", "comes_after": []}

    class PlanProvider(Scripted):
        def generate_json(self, prompt, task="extraction", schema=None):
            self.calls.append(task)
            if task == "syllabus" and fail_syllabus:
                raise RuntimeError("HTTP 429 rate limit")
            props = (schema or {}).get("properties", {})
            if "paths" in props:
                return {"paths": [], "cross_path_relations": [], "shared_foundations": []}
            if "nodes" in props:
                return {"nodes": [{"id": "S1", "parent_id": "none", "type": "study", "title": "Gradient descent", "purpose": "p", **node}],
                        "knowledge": [], "skills": [], "capabilities": [], "applications": []}
            return {}

    return PlanProvider()


def test_plan_built_during_an_outage_reports_warnings_and_regenerates_cleanly(sessions, monkeypatch):
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

    with sessions() as db:
        source, _ = _source_with_chunk(db)
        source.status = "done"
        source.profile_json = json.dumps({**GOOD_PROFILE, "schema_version": 2})
        db.commit()

    state = {"provider": _plan_provider(fail_syllabus=True)}
    monkeypatch.setattr(main, "session_scope", scope)
    monkeypatch.setattr(main, "provider", lambda: state["provider"])
    client = TestClient(main.app)

    broken = client.get("/syllabus").json()
    assert any("Could not extract topics" in w for w in broken["warnings"])
    assert any("No study topics" in w for w in broken["warnings"])
    assert not [n for n in broken["nodes"] if n["type"] == "study"]
    assert client.get("/syllabus").json()["warnings"] == broken["warnings"]  # the warnings are saved with the plan

    state["provider"] = _plan_provider(fail_syllabus=False)
    healed = client.get("/syllabus", params={"regenerate": "true"}).json()
    assert healed["warnings"] == []
    assert [n["title"] for n in healed["nodes"] if n["type"] == "study"] == ["Gradient descent"]
    assert client.get("/syllabus").json()["warnings"] == []
    with sessions() as db:
        assert db.execute(select(StudyPlan)).scalar_one().relationships_json is not None
