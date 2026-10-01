from app.db import init_db, engine, SessionLocal
from app.models import Base, Source, Job
from app.services.queue import enqueue, reclaim_interrupted


def test_reclaim_interrupted(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    # Replace the module-level engine/session objects for this isolated test.
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    test_engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(test_engine)
    TestSession = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, expire_on_commit=False)
    db = TestSession()
    src = Source(name="x", source_type="file", status="processing")
    db.add(src); db.flush(); job = Job(source_id=src.id, task="process_source", status="running", stage="embed", checkpoint="chunk:4/10")
    db.add(job); db.commit()
    assert reclaim_interrupted(db) == 1
    assert db.get(Job, job.id).status == "queued"
    db.close()
