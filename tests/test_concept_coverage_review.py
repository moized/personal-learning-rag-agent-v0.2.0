"""Regression tests for the findings of the Stage 2 adversarial review."""
import json
import re

from sqlalchemy import create_engine, func, select

from app.config import settings
from app.models import Base, Chunk, Concept, ConceptRelation, Document, Evidence, Job, MaintenanceJob, Source
from app.services import worker as worker_module
from app.services.concepts import CONCEPT_PASS, extract_batch_concepts
from app.services.knowledge_graph import extract_source_relations
from app.services.queue import enqueue
from app.services.sections import build_source_sections
from app.services.worker import Worker
from tests.test_concept_coverage import Fixed, Scripted, _chunk_ids, _pdf, _pdf_source, _sections, _source, env  # noqa: F401


def test_removal_waits_for_organize_to_pass_the_source(env):
    with env.sessions() as db:
        source = _source(db, docs=[("V", ["text"])])
        job = MaintenanceJob(task="organize_existing", status="running", progress_current=0, progress_total=1,
                             items_json=json.dumps([{"kind": "concepts", "id": source.id}]))
        db.add(job)
        db.commit()
        source_id, job_id = source.id, job.id
    assert env.client.delete(f"/sources/{source_id}").status_code == 409
    with env.sessions() as db:
        db.get(MaintenanceJob, job_id).progress_current = 1  # the organize run has passed this source
        db.commit()
    assert env.client.delete(f"/sources/{source_id}").status_code == 200


def test_backfill_yields_to_a_waiting_job_and_resumes(env, monkeypatch):
    monkeypatch.setattr(settings, "concept_batch_chunks", 1)
    with env.sessions() as db:
        source = _source(db, docs=[("V", ["Attention one.", "Attention two.", "Attention three."])])
        waiting = Source(name="new upload", source_type="file", status="queued")
        db.add(waiting)
        db.flush()
        new_job = enqueue(db, waiting.id)
        maintenance = MaintenanceJob(task="organize_existing", status="running", progress_total=1,
                                     items_json=json.dumps([{"kind": "concepts", "id": source.id}]))
        db.add(maintenance)
        db.commit()
        source_id, maintenance_id, new_job_id = source.id, maintenance.id, new_job.id
    ai = Scripted()
    monkeypatch.setattr(worker_module, "provider", lambda: ai)
    Worker()._process_maintenance_job(maintenance_id)
    with env.sessions() as db:
        job = db.get(MaintenanceJob, maintenance_id)
        assert job.status == "queued" and job.progress_current == 0
        assert ai.calls == []  # yielded before spending anything
        db.get(Job, new_job_id).status = "done"
        job.status = "running"
        db.commit()
    Worker()._process_maintenance_job(maintenance_id)
    with env.sessions() as db:
        assert db.get(MaintenanceJob, maintenance_id).status == "done"
        unread = db.scalar(select(func.count()).select_from(Chunk).join(Document).where(Document.source_id == source_id, Chunk.concept_pass < CONCEPT_PASS))
        assert unread == 0


def test_job_queue_comes_before_maintenance():
    import inspect
    loop = inspect.getsource(Worker.run)
    assert loop.index("claim_next(db)") < loop.index("_claim_maintenance_job()")


def test_a_source_removed_mid_read_is_skipped_not_failed(env, monkeypatch):
    with env.sessions() as db:
        source = _source(db, docs=[("V", ["Attention text."])])
        maintenance = MaintenanceJob(task="organize_existing", status="running", progress_total=1,
                                     items_json=json.dumps([{"kind": "concepts", "id": source.id}]))
        db.add(maintenance)
        db.commit()
        source_id, maintenance_id = source.id, maintenance.id

    class RemovesDuringCall(Scripted):
        def generate_json(self, prompt, task="extraction", schema=None):
            with env.sessions() as other:  # another request deletes the source while the AI is working
                ids = other.execute(select(Chunk.id).join(Document).where(Document.source_id == source_id)).scalars().all()
                other.query(Evidence).filter(Evidence.chunk_id.in_(ids)).delete(synchronize_session=False)
                other.query(Chunk).filter(Chunk.id.in_(ids)).delete(synchronize_session=False)
                other.query(Document).filter(Document.source_id == source_id).delete(synchronize_session=False)
                other.query(Source).filter(Source.id == source_id).delete(synchronize_session=False)
                other.commit()
            return super().generate_json(prompt, task, schema)

    monkeypatch.setattr(worker_module, "provider", lambda: RemovesDuringCall())
    Worker()._process_maintenance_job(maintenance_id)
    with env.sessions() as db:
        job = db.get(MaintenanceJob, maintenance_id)
        assert job.status == "done" and not json.loads(job.errors_json or "{}")


def test_repeated_pair_in_one_relationship_reply_merges(env):
    with env.sessions() as db:
        source = _source(db, docs=[("V", ["a", "b"])])
        ids = _chunk_ids(db, source.id)
        one, two = Concept(canonical_name="Chain Rule"), Concept(canonical_name="Backpropagation")
        db.add_all([one, two])
        db.flush()
        db.add_all([Evidence(concept_id=one.id, chunk_id=ids[0]), Evidence(concept_id=two.id, chunk_id=ids[1])])
        db.flush()
        edge = {"type": "prerequisite", "confidence": 0.9, "evidence_chunk_ids": [ids[0]], "rationale": "r"}
        reply = {"relationships": [
            {"from_concept": "Chain Rule", "to_concept": "Backpropagation", **edge},
            {"from_concept": "chain rule", "to_concept": "BACKPROPAGATION", **edge},
            {"from_concept": "Backpropagation", "to_concept": "Chain Rule", **edge},  # would close a cycle
        ]}

        class Relations(Scripted):
            def generate_json(self, prompt, task="extraction", schema=None):
                return reply

        extract_source_relations(db, Relations(), source.id)
        db.flush()
        assert db.execute(select(ConceptRelation.from_concept_id, ConceptRelation.to_concept_id)).all() == [(one.id, two.id)]


def test_relation_windows_leave_no_concept_alone_and_overlap(env):
    with env.sessions() as db:
        source = _source(db, docs=[("V", [f"t{i}" for i in range(41)])])
        for i, chunk_id in enumerate(_chunk_ids(db, source.id)):
            concept = Concept(canonical_name=f"C{i:02d}")
            db.add(concept)
            db.flush()
            db.add(Evidence(concept_id=concept.id, chunk_id=chunk_id))
        db.flush()
        ai = Scripted()
        extract_source_relations(db, ai, source.id)
        windows = [set(re.findall(r"- (C\d\d):", p)) for p, t in zip(ai.prompts, ai.calls) if t == "relationship_extraction"]
        assert len(windows) == 2 and all(2 <= len(w) <= 40 for w in windows)
        assert set().union(*windows) == {f"C{i:02d}" for i in range(41)} and windows[0] & windows[1]


def test_unusable_replies_and_offline_mode_leave_chunks_unread(env):
    from app.providers.local import LocalProvider
    with env.sessions() as db:
        source = _source(db, docs=[("V", ["Backpropagation text."])])
        ids = _chunk_ids(db, source.id)
        assert extract_batch_concepts(db, Fixed(["Backpropagation", {"concept": "x"}]), source, ids) == 0
        assert db.get(Chunk, ids[0]).concept_pass == 0
        db.commit()
        source_id = source.id
    Worker()._extract_concepts(source_id, LocalProvider())  # no AI model: sections only
    with env.sessions() as db:
        chunk = db.get(Chunk, ids[0])
        assert chunk.concept_pass == 0 and chunk.section_id is not None
    listed = env.client.get("/sources").json()[0]
    assert listed["index_pending"] == 1 and listed["section_count"] == 1


def test_concept_extraction_setting_is_respected(env, monkeypatch):
    monkeypatch.setattr(settings, "concept_extraction", False)
    with env.sessions() as db:
        source = _source(db, docs=[("V", ["Attention text."])])
        db.commit()
        source_id = source.id
    ai = Scripted()
    Worker()._extract_concepts(source_id, ai)
    assert ai.calls == []
    queued = env.client.post("/maintenance/organize-existing").json()
    with env.sessions() as db:
        assert not [i for i in json.loads(db.get(MaintenanceJob, queued["id"]).items_json) if i.get("kind") == "concepts"]
    assert env.client.get("/sources").json()[0]["index_pending"] == 0


def test_an_alias_never_takes_another_concepts_name(env):
    with env.sessions() as db:
        self_attention, attention = Concept(canonical_name="Self-Attention"), Concept(canonical_name="Attention")
        db.add_all([self_attention, attention])
        source = _source(db, docs=[("V", ["Self-attention relates tokens.", "Attention weights inputs."])])
        ids = _chunk_ids(db, source.id)
        extract_batch_concepts(db, Fixed([{"name": "Self-Attention", "description": "", "aliases": ["attention", "SELF ATTENTION", "self attention"], "chunk_ids": [ids[0]]}]), source, ids)
        assert json.loads(self_attention.aliases_json) == ["SELF ATTENTION"]
        for chunk in db.execute(select(Chunk)).scalars():
            chunk.concept_pass = 0
        extract_batch_concepts(db, Fixed([{"name": "Attention", "description": "", "chunk_ids": [ids[1]]}]), source, ids)
        db.flush()
        assert db.execute(select(Evidence.concept_id).where(Evidence.chunk_id == ids[1])).scalars().all() == [attention.id]


def test_fallback_matches_whole_words_and_odd_shapes_are_safe(env):
    with env.sessions() as db:
        source = _source(db, docs=[("V", ["Let me explain the main idea.", "AI systems learn from data.", "Item 23 here."])])
        ids = _chunk_ids(db, source.id)
        extract_batch_concepts(db, Fixed([
            {"name": "AI", "description": "", "chunk_ids": []},
            {"name": "Data", "description": "", "chunk_ids": 7, "aliases": [None, 5, "data"]},  # a scalar, not a list
            {"name": "Items", "description": "", "chunk_ids": "23"},                              # a string, not a list
            {"name": "Flags", "description": "", "chunk_ids": [True, ids[2]]},
        ]), source, ids)
        db.flush()
        links = {c.canonical_name: sorted(db.execute(select(Evidence.chunk_id).where(Evidence.concept_id == c.id)).scalars())
                 for c in db.execute(select(Concept)).scalars()}
        assert links["AI"] == [ids[1]] and links["Data"] == [ids[1]] and links["Flags"] == [ids[2]]
        assert "Items" not in links
        assert json.loads(db.execute(select(Concept).where(Concept.canonical_name == "Data")).scalar_one().aliases_json) == []


def test_retry_organization_only_repeats_failed_items(env):
    items = [{"kind": "concepts", "id": 1}, {"kind": "concepts", "id": 2}, {"kind": "source", "id": 3}]
    with env.sessions() as db:
        job = MaintenanceJob(task="organize_existing", status="failed", progress_current=3, progress_total=3,
                             items_json=json.dumps(items), errors_json=json.dumps({"concepts:2": "429"}))
        db.add(job)
        db.commit()
        job_id = job.id
    assert env.client.post(f"/maintenance/jobs/{job_id}/retry").status_code == 200
    with env.sessions() as db:
        job = db.get(MaintenanceJob, job_id)
        assert json.loads(job.items_json) == [{"kind": "concepts", "id": 2}] and job.progress_total == 1


def test_upgraded_databases_get_the_section_index(tmp_path, monkeypatch):
    from sqlalchemy import inspect, text
    import app.db as app_db
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX ix_chunks_section_id"))
    monkeypatch.setattr(app_db, "engine", engine)
    app_db._migrate_sqlite()
    assert "ix_chunks_section_id" in {index["name"] for index in inspect(engine).get_indexes("chunks")}
    engine.dispose()


def test_one_stray_chapter_line_does_not_turn_a_pdf_into_one_chapter(env, monkeypatch):
    monkeypatch.setattr(settings, "pdf_pages_per_part", 10)
    with env.sessions() as db:
        prose = _pdf_source(db, env.tmp / "prose.pdf", [f"p{i}" for i in range(16)] + ["Section 4 shows that the estimator is unbiased"] + [f"p{i}" for i in range(13)])
        build_source_sections(db, prose)
        assert [t for t, _, _ in _sections(db, prose.id)] == ["Pages 1–10", "Pages 11–20", "Pages 21–30"]
        titled = _pdf_source(db, env.tmp / "lecture.pdf", ["Lecture 3: Decision Trees Prof. Smith"] + [f"p{i}" for i in range(19)])
        build_source_sections(db, titled)
        assert [t for t, _, _ in _sections(db, titled.id)] == ["Pages 1–10", "Pages 11–20"]


def test_chapter_titles_come_from_the_pages_own_lines():
    from app.services.sections import _chapter_title
    assert _chapter_title(["Chapter 1", "Linear Regression"], "") == "Chapter 1: Linear Regression"
    assert _chapter_title(["Chapter 2: Trees and Forests", "Body text starts here"], "") == "Chapter 2: Trees and Forests"
    assert _chapter_title(["Section 4 shows that the result holds"], "") is None
    assert _chapter_title([], "Chapter 9 Anything") is None


def test_headings_survive_chunking_overlap_code_and_csharp(env, monkeypatch):
    from app.services.ingestion import chunk_text, parse_file
    monkeypatch.setattr(settings, "section_part_chunks", 2)
    markdown = env.tmp / "topic.md"
    markdown.write_text("## Topic 1\nThis paragraph sits right under its heading.", encoding="utf-8")
    pieces = chunk_text(parse_file(markdown)[0].text, 1800, 220)
    assert pieces[0].splitlines()[0] == "## Topic 1"
    code_file = env.tmp / "code.md"
    code_file.write_text("```python\nimport numpy as np\n\n# compute the mean\nx = 1\n```", encoding="utf-8")
    code = chunk_text(parse_file(code_file)[0].text, 1800, 220)
    assert not any(line.strip() == "# compute the mean" for piece in code for line in piece.splitlines())
    with env.sessions() as db:
        doc = _source(db, name="notes.md", source_type="file", docs=[("md", [
            "# Alpha\nAlpha body.", "Alpha more.\n## Beta\nBeta intro.", "## Beta\nBeta continues.", "```python\nx = 1\n# not a heading\n```",
            "## LINQ in C#\nQueries."])])
        build_source_sections(db, doc)
        assert [(t, n) for t, _, n in _sections(db, doc.id)] == [("Alpha", 1), ("Beta", 3), ("LINQ in C#", 1)]
        late = _source(db, name="late.md", source_type="file", docs=[("late", ["intro a", "intro b", "intro c", "intro d", "intro e", "# Finally\nbody"])])
        build_source_sections(db, late)
        assert [k for _, k, _ in _sections(db, late.id)] == ["part", "part", "part", "heading"]


def test_continued_slide_titles_merge_and_real_words_survive():
    from app.services.sections import CONTINUED, _clean_title
    cases = [("Backpropagation (cont’d)", "Backpropagation"), ("Gradients – continued", "Gradients"),
             ("Loss: cont.", "Loss"), ("X, continued", "X"), ("Products Discontinued", "Products Discontinued"), ("Why C#", "Why C#")]
    for title, base in cases:
        assert _clean_title(CONTINUED.sub("", title)) == base


def test_bookmark_levels_keep_all_chapters_and_ignore_useless_outlines(env):
    path = env.tmp / "book3.pdf"
    _pdf(path, 8, [("My Book", 0, [("Ch 1", 0), ("Ch 2", 2), ("Ch 3", 4)]), ("Index", 7)])
    same = env.tmp / "same.pdf"
    _pdf(same, 8, [("Ch 1", 0), ("Ch 2", 0), ("Index", 0)])
    with env.sessions() as db:
        book = _pdf_source(db, path, [f"p{i}" for i in range(8)])
        build_source_sections(db, book)
        assert _sections(db, book.id) == [("Ch 1", "chapter", 2), ("Ch 2", "chapter", 2), ("Ch 3", "chapter", 3), ("Index", "chapter", 1)]
        flat = _pdf_source(db, same, [f"p{i}" for i in range(8)])
        build_source_sections(db, flat)
        assert [k for _, k, _ in _sections(db, flat.id)] == ["pages"]
