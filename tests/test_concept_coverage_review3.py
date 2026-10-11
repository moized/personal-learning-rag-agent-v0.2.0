"""Regression tests for the third Stage 2 review."""
import json
from datetime import UTC, datetime, timedelta

from app.config import settings
from app.models import Concept, Evidence, MaintenanceJob
from app.services.ingestion import chunk_text, count_markdown_headings
from app.services.knowledge_graph import extract_source_relations
from app.services.sections import _overlap_length, _scan_chunks, build_source_sections
from tests.test_concept_coverage import Scripted, _chunk_ids, _sections, _source, env  # noqa: F401

TICKS = "`" * 3


def test_every_heading_in_a_chunk_counts(env):
    with env.sessions() as db:
        doc = _source(db, name="notes.md", source_type="file", docs=[("notes", [
            "# Machine Learning Notes\nShort intro.\n## Chapter 1: Linear Models\nLinear text.", "More linear text.", "## Chapter 2: Networks\nNet text."])])
        build_source_sections(db, doc)
        assert [(t, n) for t, _, n in _sections(db, doc.id)] == [("Machine Learning Notes", 1), ("Chapter 1: Linear Models", 1), ("Chapter 2: Networks", 1)]


def test_overlap_is_found_after_a_sentence_split():
    previous = "Intro sentence.\nAnother one. " + "x" * 50 + " The end of the old chunk."
    tail_words = previous[-settings.chunk_overlap:]
    current = " ".join(tail_words.split()) + "\nNew text."
    assert _overlap_length(previous, current) == len(" ".join(tail_words.split()))


def test_inline_triple_backticks_in_prose_do_not_open_a_fence(env):
    prose = f"To show code, put a line with `` {TICKS} `` before and after it."
    with env.sessions() as db:
        doc = _source(db, name="basics.md", source_type="file", docs=[("basics", [f"# Basics\n{prose}", "## Tables\nPipes.", "## Links\nBrackets."])])
        build_source_sections(db, doc)
        assert [t for t, _, _ in _sections(db, doc.id)] == ["Basics", "Tables", "Links"]
    assert count_markdown_headings(f"# One\n{prose}\n# Two\ntext") == 2


def test_code_comments_in_plain_text_are_not_headings(env):
    assert count_markdown_headings("import numpy as np\n# create some data\nx = 1\n# split the data\n") == 0
    with env.sessions() as db:
        lecture = _source(db, name="lecture3.txt", source_type="file", docs=[("lecture3", ["Lecture 3.\n# create some data\nx = 1", "# split the data\ny = 2"])])
        build_source_sections(db, lecture)
        assert [k for _, k, _ in _sections(db, lecture.id)] == ["document"]


def test_long_word_headings_are_kept_and_short_glued_ones_rejected(env):
    long_title = "Chapter 2: A Systematic Review of Retrieval-Augmented Generation for Learning Support"
    with env.sessions() as db:
        thesis = _source(db, name="thesis.docx", source_type="file", docs=[("thesis", ["# Chapter 1: Introduction\nText.", f"# {long_title}\nText."])])
        build_source_sections(db, thesis)
        assert [t for t, _, _ in _sections(db, thesis.id)] == ["Chapter 1: Introduction", long_title]
        legacy = _source(db, name="old.md", source_type="file", docs=[("old", ["# Week 3 The model reached 91% accuracy.", "More."])])
        build_source_sections(db, legacy)
        assert [k for _, k, _ in _sections(db, legacy.id)] == ["document"]


def test_pasted_captions_keep_their_caption_type(env, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    captions = "1\n00:00:01,000 --> 00:00:02,000\n# Upbeat music #\n\n2\n00:00:02,000 --> 00:00:03,000\n# Here comes the sun #\n\n3\n00:00:03,000 --> 00:00:04,000\nWelcome.\n"
    created = env.client.post("/sources/manual", data={"name": "talk", "payload": captions, "input_format": "srt"}).json()
    with env.sessions() as db:
        from app.models import Source
        assert db.get(Source, created["source_id"]).local_path.endswith(".srt")


def test_relation_windows_send_a_heartbeat_first(env):
    with env.sessions() as db:
        source = _source(db, docs=[("V", [f"t{i}" for i in range(60)])])
        for i, chunk_id in enumerate(_chunk_ids(db, source.id)):
            concept = Concept(canonical_name=f"C{i:02d}")
            db.add(concept)
            db.flush()
            db.add(Evidence(concept_id=concept.id, chunk_id=chunk_id))
        db.commit()
        beats = []
        extract_source_relations(db, Scripted(), source.id, before_window=lambda: beats.append(1))
        assert len(beats) == 2


def test_the_job_this_worker_runs_is_live_whatever_its_heartbeat(env, monkeypatch):
    from app import main
    from app.services.worker import Worker
    with env.sessions() as db:
        source = _source(db, docs=[("V", ["text"])])
        job = MaintenanceJob(task="organize_existing", status="running", items_json=json.dumps([{"kind": "concepts", "id": source.id}]),
                             updated_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1))
        db.add(job)
        db.commit()
        source_id, job_id = source.id, job.id
    running = Worker()
    running.current_maintenance_id = job_id
    monkeypatch.setattr(main, "worker", running)
    assert env.client.delete(f"/sources/{source_id}").status_code == 409
    running.current_maintenance_id = None
    assert env.client.delete(f"/sources/{source_id}").status_code == 200
