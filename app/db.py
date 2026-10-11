from contextlib import contextmanager
from pathlib import Path
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker
from .config import settings

url = settings.database_url
connect_args = {"check_same_thread": False, "timeout": 30.0} if url.startswith("sqlite") else {}
engine = create_engine(url, connect_args=connect_args, future=True)
if url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_busy_timeout(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

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
    source_columns = {c["name"] for c in inspect(engine).get_columns("sources")} if "sources" in inspect(engine).get_table_names() else set()
    _add_missing_columns("sources", {
        "course_id": "course_id INTEGER",
        "topic_hint": "topic_hint VARCHAR(500)",
        "description": "description TEXT",
        "metadata_json": "metadata_json TEXT",
        "profile_json": "profile_json TEXT",
        "assignment_mode": "assignment_mode VARCHAR(30) DEFAULT 'explicit_unassigned'",
        "assignment_status": "assignment_status VARCHAR(30) DEFAULT 'unassigned'",
        "assignment_reason": "assignment_reason TEXT",
    })
    _add_missing_columns("jobs", {
        "plan_json": "plan_json TEXT",
        "started_at": "started_at DATETIME",
        "error_message": "error_message VARCHAR(500)",
    })
    if "assignment_mode" not in source_columns:
        with engine.begin() as conn:
            conn.execute(text("UPDATE sources SET assignment_mode=CASE WHEN course_id IS NULL THEN 'explicit_unassigned' ELSE 'explicit_course' END"))
            conn.execute(text("UPDATE sources SET assignment_status=CASE WHEN course_id IS NULL THEN 'unassigned' ELSE 'assigned' END"))
    _add_missing_columns("groups", {"keywords_json": "keywords_json TEXT"})
    _add_missing_columns("documents", {"summary": "summary TEXT", "metadata_json": "metadata_json TEXT"})
    _add_missing_columns("chunks", {
        "parent_chunk_id": "parent_chunk_id INTEGER",
        "chunk_kind": "chunk_kind VARCHAR(20) DEFAULT 'leaf'",
        "contextual_text": "contextual_text TEXT",
        "token_estimate": "token_estimate INTEGER DEFAULT 0",
        "group_confidence": "group_confidence FLOAT",
        "section_id": "section_id INTEGER",
        "concept_pass": "concept_pass INTEGER DEFAULT 0",
    })
    with engine.begin() as conn:
        # create_all only indexes new tables; columns added above need their index explicitly.
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chunks_section_id ON chunks (section_id)"))
    _add_missing_columns("study_plans", {"relationships_json": "relationships_json TEXT"})
    _add_missing_columns("study_plan_nodes", {"path_id": "path_id VARCHAR(100) DEFAULT 'main'"})
    with engine.begin() as conn:
        conn.execute(text("UPDATE chunks SET chunk_kind='leaf' WHERE chunk_kind IS NULL"))
    _add_missing_columns("concepts", {"aliases_json": "aliases_json TEXT"})


def init_db() -> None:
    Path("./data").mkdir(exist_ok=True)
    if settings.database_url.startswith("sqlite") and ":memory:" not in settings.database_url:
        # WAL lets readers proceed while the worker writes; busy_timeout lets
        # concurrent short writes wait instead of failing immediately.
        with engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            conn.commit()
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
