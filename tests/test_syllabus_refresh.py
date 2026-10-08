from contextlib import contextmanager

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base


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

    def fake_build(db, ai, course_id):
        builds.append(course_id)
        return f"plan v{len(builds)}", []

    from app import main

    monkeypatch.setattr(main, "session_scope", scope)
    monkeypatch.setattr(main, "provider", lambda: None)
    monkeypatch.setattr(main, "build_syllabus_structured", fake_build)
    client = TestClient(main.app)

    assert client.get("/syllabus").json()["syllabus"] == "plan v1"
    assert client.get("/syllabus").json()["syllabus"] == "plan v1"  # unchanged sources: served from cache
    assert len(builds) == 1
    assert client.get("/syllabus", params={"regenerate": "true"}).json()["syllabus"] == "plan v2"
    assert client.get("/syllabus").json()["syllabus"] == "plan v2"  # the regenerated plan is what is cached now
    assert len(builds) == 2
