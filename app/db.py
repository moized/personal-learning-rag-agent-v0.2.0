from contextlib import contextmanager
from pathlib import Path
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker
from .config import settings

url = settings.database_url
connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
engine = create_engine(url, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


def _add_missing_columns(table: str, columns: dict[str, str]) -> None:
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return
    existing = {c["name"] for c in inspector.get_columns(table)}
    with engine.begin() as conn:
        for col, ddl in columns.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {ddl}"))


def _migrate_sqlite() -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    _add_missing_columns("sources", {
        "course_id": "course_id INTEGER",
        "topic_hint": "topic_hint VARCHAR(500)",
        "description": "description TEXT",
        "metadata_json": "metadata_json TEXT",
        "profile_json": "profile_json TEXT",
    })
    _add_missing_columns("jobs", {"plan_json": "plan_json TEXT"})
    _add_missing_columns("groups", {"keywords_json": "keywords_json TEXT"})
    _add_missing_columns("documents", {"summary": "summary TEXT"})
    _add_missing_columns("chunks", {
        "parent_chunk_id": "parent_chunk_id INTEGER",
        "chunk_kind": "chunk_kind VARCHAR(20) DEFAULT 'leaf'",
        "contextual_text": "contextual_text TEXT",
        "token_estimate": "token_estimate INTEGER DEFAULT 0",
        "group_confidence": "group_confidence FLOAT",
    })
    with engine.begin() as conn:
        conn.execute(text("UPDATE chunks SET chunk_kind='leaf' WHERE chunk_kind IS NULL"))
    _add_missing_columns("concepts", {"aliases_json": "aliases_json TEXT"})


def init_db() -> None:
    Path("./data").mkdir(exist_ok=True)
    from . import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _migrate_sqlite()


@contextmanager
def session_scope():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
