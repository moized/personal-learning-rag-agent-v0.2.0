from contextlib import contextmanager

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Source
from tests.plan_jobs import run_plan_jobs


def test_syllabus_is_cached_until_regenerate_is_requested(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'plan.db'}", connect_args={"check_same_thread": False})
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

    builds = []

    def fake_build(db, ai, course_id, **options):
        builds.append(course_id)
        return f"plan v{len(builds)}", []

    from app import main
    from app.services import plans, worker

    monkeypatch.setattr(main, "session_scope", scope)
    monkeypatch.setattr(worker, "session_scope", scope)
    monkeypatch.setattr(worker, "provider", lambda: None)
    monkeypatch.setattr(plans, "build_syllabus_structured", fake_build)
    with sessions() as db:
        db.add(Source(name="Lecture", source_type="transcript", status="done"))
        db.commit()
    client = TestClient(main.app)

    first = client.get("/syllabus").json()
    assert first["status"] == "updating" and builds == []  # opening the page never builds
    assert run_plan_jobs(sessions) == 1
    assert client.get("/syllabus").json()["syllabus"] == "plan v1"
    ready = client.get("/syllabus").json()
    assert ready["syllabus"] == "plan v1" and ready["status"] == "ready"  # unchanged sources: served from cache
    assert run_plan_jobs(sessions) == 0 and len(builds) == 1
    assert client.get("/syllabus", params={"regenerate": "true"}).json()["status"] == "updating"
    assert run_plan_jobs(sessions) == 1
    assert client.get("/syllabus").json()["syllabus"] == "plan v2"  # the regenerated plan is what is cached now
    assert len(builds) == 2
    engine.dispose()
