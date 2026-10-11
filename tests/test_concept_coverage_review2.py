"""Regression tests for the second Stage 2 review (problems introduced by the first fixes)."""
import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from app.config import settings
from app.models import Chunk, Concept, ConceptRelation, Document, Evidence, MaintenanceJob, Source
from app.services import worker as worker_module
from app.services.ingestion import chunk_text, parse_file
from app.services.knowledge_graph import extract_source_relations
from app.services.queue import enqueue
from app.services.sections import build_source_sections
from app.services.worker import Worker
from tests.test_concept_coverage import Scripted, _chunk_ids, _sections, _source, env  # noqa: F401


def _now():
    return datetime.now(UTC).replace(tzinfo=None)


def test_a_job_left_running_by_a_stopped_app_does_not_block_removal_forever(env):
    with env.sessions() as db:
        source = _source(db, docs=[("V", ["text"])])
        stale = MaintenanceJob(task="organize_existing", status="running", progress_current=0, progress_total=1,
                               items_json=json.dumps([{"kind": "concepts", "id": source.id}]),
                               updated_at=_now() - timedelta(seconds=settings.maintenance_stale_seconds + 60))
        db.add(stale)
        db.commit()
        source_id, stale_id = source.id, stale.id
    assert env.client.delete(f"/sources/{source_id}").status_code == 200  # a silent "running" job is not live
    with env.sessions() as db:
        db.get(MaintenanceJob, stale_id).updated_at = _now() - timedelta(seconds=settings.maintenance_stale_seconds + 60)
        fresh = MaintenanceJob(task="organize_existing", status="running", items_json="[]", updated_at=_now())
        db.add(fresh)
        db.commit()
        fresh_id = fresh.id
    assert Worker._requeue_stale_maintenance() == 1
    with env.sessions() as db:
        assert db.get(MaintenanceJob, stale_id).status == "queued" and db.get(MaintenanceJob, fresh_id).status == "running"


def test_concept_batches_refresh_the_maintenance_heartbeat(env, monkeypatch):
    with env.sessions() as db:
        source = _source(db, docs=[("V", ["Attention text."])])
        job = MaintenanceJob(task="organize_existing", status="running", items_json="[]", updated_at=_now() - timedelta(hours=1))
        db.add(job)
        db.commit()
        source_id, job_id = source.id, job.id
    Worker()._extract_concepts(source_id, Scripted(), heartbeat_job_id=job_id)
    with env.sessions() as db:
        assert db.get(MaintenanceJob, job_id).updated_at > _now() - timedelta(minutes=1)


def test_a_waiting_job_interrupts_every_kind_of_maintenance_item(env, monkeypatch):
    with env.sessions() as db:
        source = _source(db, docs=[("V", ["text"])])
        chunk_id = _chunk_ids(db, source.id)[0]
        waiting = Source(name="upload", source_type="file", status="queued")
        db.add(waiting)
        db.flush()
        enqueue(db, waiting.id)
        maintenance = MaintenanceJob(task="organize_existing", status="running", progress_total=2,
                                     items_json=json.dumps([{"kind": "chunk", "id": chunk_id}, {"kind": "source", "id": source.id}]))
        db.add(maintenance)
        db.commit()
        maintenance_id = maintenance.id
    ai = Scripted()
    monkeypatch.setattr(worker_module, "provider", lambda: ai)
    Worker()._process_maintenance_job(maintenance_id)
    with env.sessions() as db:
        job = db.get(MaintenanceJob, maintenance_id)
        assert job.status == "queued" and job.progress_current == 0 and ai.calls == []


def test_existing_chunks_are_never_rechunked(env, tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Intro\nOld paragraph one.\n\n# Next\nOld paragraph two.", encoding="utf-8")
    with env.sessions() as db:
        source = Source(name="notes.md", source_type="file", local_path=str(path), status="processing")
        db.add(source)
        db.flush()
        doc = Document(source_id=source.id, external_id=path.name, title="notes")
        db.add(doc)
        db.flush()
        db.add(Chunk(document_id=doc.id, chunk_index=0, chunk_kind="leaf", text="chunked by an older version"))
        db.flush()
        Worker()._ensure_file_documents(db, source)
        leaves = db.execute(select(Chunk.text).where(Chunk.document_id == doc.id, Chunk.chunk_kind == "leaf")).scalars().all()
        assert leaves == ["chunked by an older version"]


def test_headings_are_structure_only_in_markdown_like_text(env, tmp_path):
    srt = tmp_path / "talk.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n# Upbeat music #\n\n2\n00:00:02,000 --> 00:00:03,000\nWelcome everyone.\n", encoding="utf-8")
    one_hash = tmp_path / "data.txt"
    one_hash.write_text("Results\n# of samples 12\nmore text", encoding="utf-8")
    notes = tmp_path / "notes.txt"
    notes.write_text("# Intro\nHello.\n# Method\nSteps.", encoding="utf-8")
    assert "\n\n# of samples" not in parse_file(one_hash)[0].text
    assert parse_file(notes)[0].text.count("\n\n") >= 2
    with env.sessions() as db:
        captions = _source(db, name="talk.srt", source_type="file", docs=[("talk", [chunk_text(parse_file(srt)[0].text)[0]])])
        build_source_sections(db, captions)
        assert [k for _, k, _ in _sections(db, captions.id)] == ["document"]


def test_a_heading_ending_a_chunk_opens_the_next_one(env):
    with env.sessions() as db:
        doc = _source(db, name="notes.md", source_type="file", docs=[("notes", [
            "# Alpha\nAlpha body.", "More alpha text.\n## Summary", "Summary body here."])])
        build_source_sections(db, doc)
        assert [(t, n) for t, _, n in _sections(db, doc.id)] == [("Alpha", 2), ("Summary", 1)]


def test_code_fences_are_tracked_across_chunks(env):
    with env.sessions() as db:
        doc = _source(db, name="notes.md", source_type="file", docs=[("notes", [
            "# Intro\nSome text.\n```python\nimport x", "# Load the raw survey data df = read()\n```\nAfter the code.", "## Results\nNumbers."])])
        build_source_sections(db, doc)
        assert [(t, n) for t, _, n in _sections(db, doc.id)] == [("Intro", 2), ("Results", 1)]


def test_run_on_titles_from_older_chunks_are_not_headings():
    from app.services.sections import _heading_title
    assert _heading_title("## Topic 0 Of the gamma gradient of sample and of beta alpha sample loss the delta network") is None
    assert _heading_title("## Results. these numbers show that") is None
    assert _heading_title("## 1. Introduction") == "1. Introduction"


def test_relation_windows_commit_as_they_go(env, monkeypatch):
    with env.sessions() as db:
        source = _source(db, docs=[("V", [f"t{i}" for i in range(60)])])
        ids = _chunk_ids(db, source.id)
        for i, chunk_id in enumerate(ids):
            concept = Concept(canonical_name=f"C{i:02d}")
            db.add(concept)
            db.flush()
            db.add(Evidence(concept_id=concept.id, chunk_id=chunk_id))
        db.commit()
        source_id = source.id

    class FailsSecondWindow(Scripted):
        def generate_json(self, prompt, task="extraction", schema=None):
            self.calls.append(task)
            if len(self.calls) == 2:
                raise RuntimeError("HTTP 429 rate limit")
            return {"relationships": [{"from_concept": "C00", "to_concept": "C01", "type": "related", "confidence": 0.9,
                                       "evidence_chunk_ids": [ids[0]], "rationale": "r"}]}

    with env.sessions() as db:
        with pytest.raises(RuntimeError):
            extract_source_relations(db, FailsSecondWindow(), source_id)
        db.rollback()
    with env.sessions() as db:
        assert db.scalar(select(func.count()).select_from(ConceptRelation)) == 1  # window 1 survived


def test_tiny_batch_sizes_never_leave_a_concept_alone(env, monkeypatch):
    monkeypatch.setattr(settings, "relationship_batch_size", 3)
    with env.sessions() as db:
        source = _source(db, docs=[("V", [f"t{i}" for i in range(7)])])
        for i, chunk_id in enumerate(_chunk_ids(db, source.id)):
            concept = Concept(canonical_name=f"K{i}")
            db.add(concept)
            db.flush()
            db.add(Evidence(concept_id=concept.id, chunk_id=chunk_id))
        db.commit()
        ai = Scripted()
        extract_source_relations(db, ai, source.id)
        import re
        windows = [set(re.findall(r"- (K\d):", p)) for p, t in zip(ai.prompts, ai.calls) if t == "relationship_extraction"]
        assert all(len(w) >= 2 for w in windows) and set().union(*windows) == {f"K{i}" for i in range(7)}


def test_library_separates_sections_and_key_ideas_pending(env, monkeypatch):
    with env.sessions() as db:
        _source(db, docs=[("V", ["text"])])
        db.commit()
    listed = env.client.get("/sources").json()[0]
    assert listed["sections_pending"] == 1 and listed["concepts_pending"] == 1
    monkeypatch.setattr(settings, "concept_extraction", False)
    listed = env.client.get("/sources").json()[0]
    assert listed["sections_pending"] == 1 and listed["concepts_pending"] == 0
