import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models import Base, Chunk, Course, Document, Job, Source
from app.services import ingestion
from app.services import worker as worker_module
from app.services.ingestion import TranscriptUnavailable, fetch_youtube_transcript, list_youtube_entries
from app.services.queue import enqueue
from app.services.retrieval import collapse_into_parents
from app.services.worker import NO_TEXT_MESSAGES, Worker, concise_space_name

PLAYLIST = "https://www.youtube.com/playlist?list=PLtest"
TEXT = "Gradient descent lowers the loss step by step.\n\nThe learning rate sets the step size."


class QuietProvider:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def generate_json(self, prompt, task="extraction", schema=None):
        return {}

    def generate_text(self, prompt, task="study"):
        return ""


@pytest.fixture()
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'ingest.db'}", connect_args={"check_same_thread": False})
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

    from app import main
    monkeypatch.setattr(worker_module, "session_scope", scope)
    monkeypatch.setattr(main, "session_scope", scope)
    monkeypatch.setattr(worker_module, "provider", lambda: QuietProvider())
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    yield SimpleNamespace(sessions=sessions, client=TestClient(main.app), tmp=tmp_path)
    engine.dispose()


def _videos(*ids):
    return [{"id": vid, "title": f"Lecture {vid}", "channel": "MIT", "upload_date": "2024-01-15", "duration_seconds": 3600} for vid in ids]


def _youtube_job(env, name=PLAYLIST, uri=PLAYLIST):
    with env.sessions() as db:
        source = Source(name=name, source_type="youtube", uri=uri, status="queued")
        db.add(source)
        db.flush()
        job = enqueue(db, source.id)
        db.commit()
        return source.id, job.id


def _transcripts(monkeypatch, behaviour):
    calls = []

    def fetch(video_id):
        calls.append(video_id)
        outcome = behaviour.get(video_id, TEXT)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(worker_module, "fetch_youtube_transcript", fetch)
    return calls


def _docs(env, source_id):
    with env.sessions() as db:
        docs = db.execute(select(Document).where(Document.source_id == source_id).order_by(Document.id)).scalars().all()
        indexed = {doc.id for doc in docs if db.execute(select(Chunk.id).where(Chunk.document_id == doc.id, Chunk.chunk_kind == "leaf")).first()}
        return {doc.external_id: (doc.status, doc.error, doc.id in indexed, json.loads(doc.metadata_json or "{}")) for doc in docs}


def test_playlist_continues_past_a_video_without_captions(env, monkeypatch):
    monkeypatch.setattr(worker_module, "list_youtube_entries", lambda url: ({"kind": "playlist", "title": "Hands-On Deep Learning", "channel": "MIT"}, _videos("v1", "v2", "v3", "v4")))
    _transcripts(monkeypatch, {"v2": TranscriptUnavailable("Transcript unavailable for v2", "This video has no captions.")})
    source_id, job_id = _youtube_job(env)
    Worker().process_job(job_id)

    docs = _docs(env, source_id)
    assert docs["v2"][:3] == ("needs_input", "This video has no captions.", False)
    assert all(docs[v][0] == "done" and docs[v][2] for v in ("v1", "v3", "v4"))
    assert docs["v1"][3]["upload_date"] == "2024-01-15" and docs["v1"][3]["channel"] == "MIT"
    with env.sessions() as db:
        source = db.get(Source, source_id)
        assert source.status == "done"
        assert source.name == "Hands-On Deep Learning"  # the URL placeholder is replaced by the real title
        assert json.loads(source.metadata_json)["platform"]["video_count"] == 4
        assert db.get(Job, job_id).status == "done"


def test_temporary_youtube_block_stops_fetching_but_keeps_finished_videos(env, monkeypatch):
    monkeypatch.setattr(worker_module, "list_youtube_entries", lambda url: ({"kind": "playlist", "title": "Course"}, _videos("v1", "v2", "v3", "v4")))
    calls = _transcripts(monkeypatch, {"v2": RuntimeError("Transcript unavailable for v2: IpBlocked")})
    source_id, job_id = _youtube_job(env)
    Worker().process_job(job_id)

    assert calls == ["v1", "v2"]  # no further requests while YouTube is refusing them
    docs = _docs(env, source_id)
    assert docs["v1"][0] == "done" and docs["v1"][2]
    assert [docs[v][0] for v in ("v2", "v3", "v4")] == ["failed", "failed", "failed"]
    with env.sessions() as db:
        assert db.get(Source, source_id).status == "done"


def test_block_before_any_video_succeeds_makes_the_job_retry(env, monkeypatch):
    monkeypatch.setattr(worker_module, "list_youtube_entries", lambda url: ({"kind": "playlist", "title": "Course"}, _videos("v1", "v2")))
    _transcripts(monkeypatch, {"v1": RuntimeError("Transcript unavailable for v1: RequestBlocked")})
    source_id, job_id = _youtube_job(env)
    worker = Worker()
    with pytest.raises(RuntimeError) as caught:
        worker.process_job(job_id)
    assert worker._record_failure(job_id, caught.value) == 1
    with env.sessions() as db:
        assert db.get(Job, job_id).status == "retrying"
        assert "captions" in db.get(Job, job_id).error_message


def test_video_without_captions_needs_input_instead_of_ready(env, monkeypatch):
    url = "https://www.youtube.com/watch?v=BaW_jenozKc"
    monkeypatch.setattr(worker_module, "list_youtube_entries", lambda u: ({"kind": "video", "title": "A talk"}, _videos("BaW_jenozKc")))
    _transcripts(monkeypatch, {"BaW_jenozKc": TranscriptUnavailable("Transcript unavailable", "Captions are turned off for this video.")})
    source_id, job_id = _youtube_job(env, name=url, uri=url)
    Worker().process_job(job_id)
    with env.sessions() as db:
        source = db.get(Source, source_id)
        assert source.status == "needs_input"
        assert source.error == NO_TEXT_MESSAGES["youtube"]
        assert db.get(Job, job_id).status == "done"
    listed = env.client.get("/sources").json()[0]
    assert listed["status"] == "needs_input" and listed["status_message"] == NO_TEXT_MESSAGES["youtube"]
    assert listed["issues"][0]["message"] == "Captions are turned off for this video."


def test_file_without_readable_text_needs_input(env):
    empty = env.tmp / "uploads" / "scanned.txt"
    empty.write_text("   \n\n ", encoding="utf-8")
    with env.sessions() as db:
        source = Source(name="scanned.pdf", source_type="file", local_path=str(empty), status="queued")
        db.add(source)
        db.flush()
        job = enqueue(db, source.id)
        db.commit()
        source_id, job_id = source.id, job.id
    Worker().process_job(job_id)
    with env.sessions() as db:
        source = db.get(Source, source_id)
        assert source.status == "needs_input" and source.error == NO_TEXT_MESSAGES["default"]


def _needs_input_video(env, *, with_second=False):
    with env.sessions() as db:
        source = Source(name="Course", source_type="youtube", uri=PLAYLIST, status="done")
        db.add(source)
        db.flush()
        missing = Document(source_id=source.id, external_id="v2", title="Lecture v2", status="needs_input", error="This video has no captions.")
        db.add(missing)
        other = None
        if with_second:
            other = Document(source_id=source.id, external_id="v3", title="Lecture v3", status="needs_input", error="This video has no captions.")
            db.add(other)
        db.commit()
        return source.id, missing.id, other.id if other else None


def test_pasted_transcript_is_indexed_without_refetching_other_videos(env, monkeypatch):
    source_id, doc_id, other_id = _needs_input_video(env, with_second=True)
    srt = "1\n00:00:01,000 --> 00:00:04,000\nGradient descent lowers the loss.\n\n2\n00:00:04,000 --> 00:00:08,000\nThe learning rate sets the step size.\n"
    response = env.client.post(f"/documents/{doc_id}/transcript", json={"text": srt, "input_format": "srt"})
    assert response.status_code == 200
    assert env.client.post(f"/documents/{doc_id}/transcript", json={"text": "again"}).status_code == 409  # job still active

    monkeypatch.setattr(worker_module, "list_youtube_entries", lambda url: ({"kind": "playlist", "title": "Course"}, _videos("v2", "v3")))
    calls = _transcripts(monkeypatch, {})
    Worker().process_job(response.json()["job_id"])
    assert calls == []  # pasted video has text; the other needs-input video is not retried silently

    docs = _docs(env, source_id)
    assert docs["v2"][0] == "done" and docs["v2"][2] and docs["v2"][3]["transcript_origin"] == "pasted"
    assert docs["v3"][0] == "needs_input"
    with env.sessions() as db:
        text = " ".join(c.text for c in db.execute(select(Chunk).where(Chunk.document_id == doc_id)).scalars())
        assert "-->" not in text and "learning rate" in text
        assert db.get(Source, source_id).status == "done"
    assert env.client.post(f"/documents/{doc_id}/transcript", json={"text": "again"}).status_code == 409  # already has text


def test_retry_resets_unreadable_parts_and_refuses_while_processing(env):
    source_id, doc_id, _ = _needs_input_video(env)
    first = env.client.post(f"/sources/{source_id}/reprocess")
    assert first.status_code == 200 and first.json()["documents_retried"] == 1
    assert env.client.post(f"/sources/{source_id}/reprocess").status_code == 409
    with env.sessions() as db:
        assert db.get(Document, doc_id).status == "pending"
        assert db.get(Source, source_id).status == "queued"


def test_pasting_text_for_a_source_without_text(env):
    old = env.tmp / "uploads" / "scan.pdf"
    old.write_bytes(b"%PDF-1.4 image only")
    with env.sessions() as db:
        empty = Source(name="scan.pdf", source_type="file", local_path=str(old), content_hash="old", status="needs_input", error="no text")
        taken = Source(name="notes", source_type="transcript", content_hash=None, status="done")
        video = Source(name="Course", source_type="youtube", uri=PLAYLIST, status="needs_input")
        db.add_all([empty, taken, video])
        db.commit()
        empty_id, video_id = empty.id, video.id

    assert env.client.post(f"/sources/{video_id}/text", json={"text": "x"}).status_code == 409
    assert env.client.post(f"/sources/{empty_id}/text", json={"text": "   "}).status_code == 400
    response = env.client.post(f"/sources/{empty_id}/text", json={"text": TEXT})
    assert response.status_code == 200
    with env.sessions() as db:
        source = db.get(Source, empty_id)
        assert source.status == "queued" and source.error is None
        assert source.local_path.endswith("scan.txt") and open(source.local_path, encoding="utf-8").read() == TEXT
    assert not old.exists()  # the unreadable upload is replaced, not left behind

    with env.sessions() as db:
        db.get(Job, response.json()["job_id"]).status = "done"
        other = Source(name="other.pdf", source_type="file", status="needs_input")
        db.add(other)
        db.commit()
        other_id = other.id
    duplicate = env.client.post(f"/sources/{other_id}/text", json={"text": TEXT})
    assert duplicate.status_code == 409 and "scan.pdf" in duplicate.json()["detail"]


def test_sources_endpoint_shows_space_and_why(env):
    with env.sessions() as db:
        course = Course(name="Deep Learning")
        db.add(course)
        db.flush()
        db.add(Source(name="Lecture", source_type="transcript", status="done", course_id=course.id,
                      assignment_status="assigned", assignment_reason='Filed into "Deep Learning" automatically (82% match).'))
        db.commit()
    listed = env.client.get("/sources").json()[0]
    assert listed["course_name"] == "Deep Learning"
    assert listed["assignment_reason"].startswith("Filed into")
    assert listed["status_message"] is None and listed["issues"] == []


def test_leaves_inside_a_selected_parent_window_are_not_repeated():
    parent = SimpleNamespace(id=10, chunk_kind="parent", parent_chunk_id=None)
    other_parent_leaf = SimpleNamespace(id=3, chunk_kind="leaf", parent_chunk_id=20)
    leaf_a = SimpleNamespace(id=1, chunk_kind="leaf", parent_chunk_id=10)
    leaf_b = SimpleNamespace(id=2, chunk_kind="leaf", parent_chunk_id=10)
    collapsed = collapse_into_parents([(leaf_a, 0.9), (other_parent_leaf, 0.8), (parent, 0.81), (leaf_b, 0.7)])
    assert [(chunk.id, score) for chunk, score in collapsed] == [(10, 0.9), (3, 0.8)]


def _transcript_api(monkeypatch, fetch=None, listing=None):
    from youtube_transcript_api import YouTubeTranscriptApi
    if fetch:
        monkeypatch.setattr(YouTubeTranscriptApi, "fetch", fetch)
    if listing is not None:
        monkeypatch.setattr(YouTubeTranscriptApi, "list", lambda self, video_id: listing)


def _named_error(name):
    return type(name, (Exception,), {})("details from the library")


def test_disabled_captions_are_reported_as_needing_a_transcript(monkeypatch):
    def fetch(self, video_id, languages):
        raise _named_error("TranscriptsDisabled")
    _transcript_api(monkeypatch, fetch=fetch)
    with pytest.raises(TranscriptUnavailable) as caught:
        fetch_youtube_transcript("BaW_jenozKc")
    assert caught.value.user_message == "Captions are turned off for this video."


def test_other_languages_are_used_when_english_and_turkish_are_missing(monkeypatch):
    def fetch(self, video_id, languages):
        raise _named_error("NoTranscriptFound")
    snippet = lambda text: SimpleNamespace(text=text)
    generated = SimpleNamespace(is_generated=True, fetch=lambda: [snippet("auto")])
    manual = SimpleNamespace(is_generated=False, fetch=lambda: [snippet("Bonjour"), snippet("tout le monde")])
    _transcript_api(monkeypatch, fetch=fetch, listing=[generated, manual])
    assert fetch_youtube_transcript("BaW_jenozKc") == "Bonjour tout le monde"


def test_no_captions_in_any_language_needs_a_transcript(monkeypatch):
    def fetch(self, video_id, languages):
        raise _named_error("NoTranscriptFound")
    _transcript_api(monkeypatch, fetch=fetch, listing=[])
    with pytest.raises(TranscriptUnavailable):
        fetch_youtube_transcript("BaW_jenozKc")


def test_blocked_requests_stay_retryable(monkeypatch):
    def fetch(self, video_id, languages):
        raise _named_error("IpBlocked")
    _transcript_api(monkeypatch, fetch=fetch)
    with pytest.raises(RuntimeError) as caught:
        fetch_youtube_transcript("BaW_jenozKc")
    assert not isinstance(caught.value, TranscriptUnavailable)


class _FakeYDL:
    info: dict | Exception = {}

    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        if isinstance(self.info, Exception):
            raise self.info
        return self.info


def test_single_video_metadata_and_fallback(monkeypatch):
    import yt_dlp
    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYDL)
    _FakeYDL.info = {"title": "Attention Is All You Need, explained", "channel": "Lab", "upload_date": "20240115", "duration": 3600.0}
    info, videos = list_youtube_entries("https://youtu.be/BaW_jenozKc?si=tracking")
    assert info == {"kind": "video", "title": "Attention Is All You Need, explained", "channel": "Lab", "upload_date": "2024-01-15"}
    assert videos[0]["id"] == "BaW_jenozKc" and videos[0]["duration_seconds"] == 3600

    _FakeYDL.info = RuntimeError("network down")
    info, videos = list_youtube_entries("https://youtu.be/BaW_jenozKc")
    assert videos[0]["title"] == "BaW_jenozKc" and info["title"] == ""


def test_playlist_metadata(monkeypatch):
    import yt_dlp
    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYDL)
    _FakeYDL.info = {"title": "Hands-On Deep Learning", "channel": "MIT", "entries": [{"id": "a", "title": "1: Intro", "duration": 100}, None]}
    info, videos = list_youtube_entries(PLAYLIST)
    assert info["title"] == "Hands-On Deep Learning" and [v["title"] for v in videos] == ["1: Intro"]
    assert ingestion.list_youtube_videos(PLAYLIST) == [("a", "1: Intro")]


class _CapturingProvider(QuietProvider):
    def __init__(self, answers=None):
        self.prompts, self.answers = [], answers or {}

    def generate_json(self, prompt, task="extraction", schema=None):
        self.prompts.append(prompt)
        return self.answers.get(task, {})


def test_profile_uses_the_playlist_title_not_the_first_video(env):
    with env.sessions() as db:
        playlist = Source(name="MIT 15.773 Hands-On Deep Learning", source_type="youtube", uri=PLAYLIST, status="processing")
        video = Source(name="https://youtu.be/BaW_jenozKc", source_type="youtube", uri="https://www.youtube.com/watch?v=BaW_jenozKc", status="processing")
        db.add_all([playlist, video])
        db.flush()
        for source, titles in ((playlist, ["1: Introduction to Neural Networks", "2: Training"]), (video, ["Attention explained"])):
            for index, title in enumerate(titles):
                doc = Document(source_id=source.id, external_id=f"{source.id}-{index}", title=title)
                db.add(doc)
                db.flush()
                db.add(Chunk(document_id=doc.id, chunk_index=0, chunk_kind="leaf", text="text"))
        db.flush()
        provider = _CapturingProvider()
        Worker()._ensure_source_profile(db, playlist, provider)
        Worker()._ensure_source_profile(db, video, provider)
        assert "Title: MIT 15.773 Hands-On Deep Learning" in provider.prompts[0]
        assert "Title: Attention explained" in provider.prompts[1]


def test_new_space_names_are_short_and_the_reason_is_recorded(env):
    assert concise_space_name("1: Introduction to Neural Networks and Deep Learning; Training Deep NNs") == "Introduction to Neural Networks and Deep Learning; Training"
    assert concise_space_name("Lecture 3 - Backpropagation") == "Backpropagation"
    with env.sessions() as db:
        source = Source(name="Lecture", source_type="transcript", status="processing", assignment_mode="auto",
                        profile_json=json.dumps({"course_title": "", "suggested_group_names": ["Deep Learning"],
                                                 "primary_learning_goal": "Understand how neural networks are trained end to end"}))
        db.add(source)
        db.flush()
        Worker()._auto_group_source(db, source, QuietProvider())
        assert db.get(Course, source.course_id).name == "Deep Learning"
        assert source.assignment_reason.startswith('Created a new space "Deep Learning"')

        second = Source(name="More", source_type="transcript", status="processing", assignment_mode="auto", profile_json="{}")
        db.add(second)
        db.flush()
        chooser = _CapturingProvider({"classification": {"course_id": source.course_id, "new_course_name": "", "confidence": 0.82,
                                                         "reason": "Both cover training neural networks."}})
        Worker()._auto_group_source(db, second, chooser)
        assert second.course_id == source.course_id
        assert second.assignment_reason == 'Filed into "Deep Learning" automatically (82% match). Both cover training neural networks.'
