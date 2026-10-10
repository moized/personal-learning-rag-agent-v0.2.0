from contextlib import contextmanager

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Chunk, Document, Job, Source
from app.providers.base import ProviderAuthenticationError
from app.services import worker as worker_module
from app.services.worker import Worker
from app.services.queue import claim_next


def _database(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'youtube.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

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
    return engine, sessions, scope


def test_youtube_source_endpoint_creates_durable_job(tmp_path, monkeypatch):
    engine, sessions, scope = _database(tmp_path, monkeypatch)
    from app import main

    monkeypatch.setattr(main, "session_scope", scope)
    response = TestClient(main.app).post(
        "/sources/youtube",
        params={"url": "https://www.youtube.com/watch?v=BaW_jenozKc"},
    )
    assert response.status_code == 200
    created = response.json()
    assert created["status"] == "queued"
    duplicate = TestClient(main.app).post(
        "/sources/youtube", params={"url": "https://youtu.be/BaW_jenozKc?si=tracking"}
    ).json()
    assert duplicate["deduplicated"] is True
    assert duplicate["source_id"] == created["source_id"]
    assert duplicate["job_id"] == created["job_id"]
    with sessions() as db:
        source = db.get(Source, created["source_id"])
        job = db.get(Job, created["job_id"])
        assert source.uri == "https://www.youtube.com/watch?v=BaW_jenozKc"
        assert source.source_type == "youtube"
        assert job.status == "queued"
        assert job.task == "process_source"
        claimed = claim_next(db)
        assert claimed.id == job.id
        assert claimed.status == "running"
        assert claimed.started_at is not None
    engine.dispose()


def test_retry_endpoint_preserves_checkpoint_and_resets_failed_status(tmp_path, monkeypatch):
    engine, sessions, scope = _database(tmp_path, monkeypatch)
    from app import main

    monkeypatch.setattr(main, "session_scope", scope)
    with sessions() as db:
        source = Source(name="video", source_type="youtube", status="failed", error="old detail")
        db.add(source)
        db.flush()
        job = Job(
            source_id=source.id, task="process_source", status="failed", stage="index",
            checkpoint="chunk:4/75", progress_current=4, progress_total=75,
            retry_count=1, error="old detail", error_message="Previous attempt failed.",
        )
        db.add(job)
        db.commit()
        job_id = job.id

    response = TestClient(main.app).post(f"/jobs/{job_id}/retry")
    assert response.status_code == 200
    assert response.json()["checkpoint"] == "chunk:4/75"
    with sessions() as db:
        job = db.get(Job, job_id)
        source = db.get(Source, job.source_id)
        assert job.status == "queued"
        assert job.stage == "index"
        assert job.checkpoint == "chunk:4/75"
        assert job.progress_current == 4
        assert job.retry_count == 0
        assert job.error is None and job.error_message is None
        assert source.status == "queued" and source.error is None
    engine.dispose()


def test_youtube_extraction_failure_is_persisted_and_returned_safely(tmp_path, monkeypatch):
    engine, sessions, scope = _database(tmp_path, monkeypatch)
    from app import main

    monkeypatch.setattr(main, "session_scope", scope)
    created = TestClient(main.app).post(
        "/sources/youtube", params={"url": "https://www.youtube.com/watch?v=BaW_jenozKc"}
    ).json()
    monkeypatch.setattr(worker_module.settings, "retry_max", 0)

    with sessions() as db:
        job = db.get(Job, created["job_id"])
        job.status = "running"
        job.stage = "transcript"
        db.commit()

    Worker()._record_failure(created["job_id"], RuntimeError("network traceback and private details"))

    with sessions() as db:
        job = db.get(Job, created["job_id"])
        source = db.get(Source, created["source_id"])
        assert job.status == "failed"
        assert job.retry_count == 1
        assert job.error == "network traceback and private details"
        assert "captions" in job.error_message
        assert source.status == "failed"
    public_job = TestClient(main.app).get("/jobs").json()[0]
    assert public_job["status"] == "failed"
    assert "captions" in public_job["error_message"]
    assert "traceback" not in str(public_job)
    engine.dispose()


def test_retryable_failure_is_visible_as_retrying(tmp_path, monkeypatch):
    engine, sessions, _ = _database(tmp_path, monkeypatch)
    monkeypatch.setattr(worker_module.settings, "retry_max", 4)
    with sessions() as db:
        source = Source(name="video", source_type="youtube", status="processing")
        db.add(source)
        db.flush()
        job = Job(source_id=source.id, task="process_source", status="running", stage="transcript")
        db.add(job)
        db.commit()
        job_id = job.id

    assert Worker()._record_failure(job_id, RuntimeError("temporary network failure")) == 1
    with sessions() as db:
        job = db.get(Job, job_id)
        assert job.status == "retrying"
        assert job.error_message is not None
    engine.dispose()


def test_invalid_provider_credentials_fail_without_retry(tmp_path, monkeypatch):
    engine, sessions, _ = _database(tmp_path, monkeypatch)
    monkeypatch.setattr(worker_module.settings, "retry_max", 4)
    with sessions() as db:
        source = Source(name="video", source_type="youtube", status="processing")
        db.add(source)
        db.flush()
        job = Job(source_id=source.id, task="process_source", status="running", stage="index")
        db.add(job)
        db.commit()
        job_id = job.id

    assert Worker()._record_failure(job_id, ProviderAuthenticationError("Gemini authentication failed (401)")) is None
    with sessions() as db:
        job = db.get(Job, job_id)
        assert job.status == "failed"
        assert job.retry_count == 1
        assert "GEMINI_API_KEY" in job.error_message
    engine.dispose()


def test_successful_worker_run_marks_job_done(tmp_path, monkeypatch):
    engine, sessions, _ = _database(tmp_path, monkeypatch)
    with sessions() as db:
        source = Source(name="local", source_type="file", status="queued")
        db.add(source)
        db.flush()
        # An already-indexed chunk: a source with no text now finishes as needs_input.
        document = Document(source_id=source.id, external_id="local", title="local", status="processing")
        db.add(document)
        db.flush()
        db.add(Chunk(document_id=document.id, chunk_index=0, chunk_kind="leaf", text="Indexed text.",
                     embedding_status="done", classification_status="done"))
        job = Job(source_id=source.id, task="process_source", status="running")
        db.add(job)
        db.commit()
        job_id = job.id

    class DummyProvider:
        pass

    monkeypatch.setattr(worker_module, "provider", lambda: DummyProvider())
    monkeypatch.setattr(Worker, "_ensure_file_documents", lambda *args: None)
    monkeypatch.setattr(Worker, "_ensure_source_profile", lambda *args: None)
    Worker().process_job(job_id)

    with sessions() as db:
        job = db.get(Job, job_id)
        source = db.get(Source, job.source_id)
        assert job.status == "done"
        assert job.stage == "done"
        assert job.checkpoint == "done"
        assert source.status == "done"
    engine.dispose()


def test_index_failure_does_not_reuse_transcript_progress(tmp_path, monkeypatch):
    engine, sessions, _ = _database(tmp_path, monkeypatch)
    with sessions() as db:
        source = Source(name="video", source_type="youtube", uri="https://www.youtube.com/watch?v=BaW_jenozKc")
        db.add(source)
        db.flush()
        document = Document(source_id=source.id, external_id="BaW_jenozKc", title="video", status="pending")
        db.add(document)
        db.flush()
        chunk = Chunk(document_id=document.id, chunk_index=0, chunk_kind="leaf", text="A transcript chunk.")
        db.add(chunk)
        job = Job(source_id=source.id, task="process_source", status="running")
        db.add(job)
        db.commit()
        job_id = job.id

    class FailingProvider:
        def embed(self, texts):
            raise ProviderAuthenticationError("Gemini authentication failed (401)")

        def generate_json(self, *args, **kwargs):
            return {"domain": "", "topics": [], "level": "", "description": "", "suggested_group_names": []}

    monkeypatch.setattr(worker_module, "provider", lambda: FailingProvider())
    monkeypatch.setattr(Worker, "_ensure_youtube_documents", lambda self, db, src: 1)
    import pytest
    with pytest.raises(ProviderAuthenticationError):
        Worker().process_job(job_id)

    with sessions() as db:
        job = db.get(Job, job_id)
        assert job.stage == "index"
        assert job.progress_current == 0
        assert job.progress_total == 1
    engine.dispose()
