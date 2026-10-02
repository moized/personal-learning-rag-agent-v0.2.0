import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
from sqlalchemy import select, or_
from .config import settings
from .db import init_db, session_scope
from .models import Source, Job, Group, Course, Chunk, Document, Concept, ConceptRelation, Evidence, AgentTrace
from .services.files import save_upload, save_text_content
from .services.queue import enqueue
from .services.worker import Worker
from .providers.factory import provider
from .services.study import answer
from .services.curriculum import build_syllabus, build_study_pack_prompt
from .services.ingestion import parse_structured_literal, clean_caption_text

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
worker: Worker | None = None


class StudyRequest(BaseModel):
    question: str
    group_id: int | None = None
    course_id: int | None = None
    top_k: int = Field(default=8, ge=1, le=50)


class StudyPackRequest(BaseModel):
    topic: str
    course_id: int | None = None
    top_k: int = Field(default=12, ge=1, le=30)


class CourseRequest(BaseModel):
    name: str
    description: str | None = None
    domain: str | None = None
    outline: str | None = None


class StructuredSourceRequest(BaseModel):
    name: str
    course_id: int | None = None
    domain_hint: str | None = None
    topic_hint: str | None = None
    description: str | None = None
    metadata: dict | None = None
    transcript: str | None = None
    content: str | None = None


@app.on_event("startup")
def startup():
    global worker
    init_db()
    worker = Worker()
    worker.start()


@app.on_event("shutdown")
def shutdown():
    if worker:
        worker.stop()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm": bool(settings.gemini_api_key),
        "reasoning_model": settings.gemini_reasoning_model,
        "fast_model": settings.gemini_fast_model,
        "embedding_model": settings.gemini_embedding_model,
        "retrieval": (
            "adaptive multi-query dense + BM25 + RRF + prerequisite graph + optional LLM rerank "
            "+ parent/neighbor context"
        ),
    }


@app.post("/courses")
def create_course(req: CourseRequest):
    with session_scope() as db:
        existing = db.execute(select(Course).where(Course.name == req.name)).scalar_one_or_none()
        if existing:
            return {"id": existing.id, "name": existing.name, "deduplicated": True}
        course = Course(name=req.name, description=req.description, domain=req.domain, outline=req.outline)
        db.add(course)
        db.flush()
        return {"id": course.id, "name": course.name}


@app.get("/courses")
def courses():
    with session_scope() as db:
        return [
            {"id": c.id, "name": c.name, "description": c.description, "domain": c.domain, "outline": c.outline}
            for c in db.execute(select(Course).order_by(Course.name)).scalars().all()
        ]


@app.get("/sources")
def sources(course_id: int | None = None):
    with session_scope() as db:
        query = select(Source).order_by(Source.created_at.desc())
        if course_id is not None:
            query = query.where(Source.course_id == course_id)
        return [
            {
                "id": source.id,
                "name": source.name,
                "source_type": source.source_type,
                "course_id": source.course_id,
                "status": source.status,
                "topic_hint": source.topic_hint,
                "domain_hint": source.domain_hint,
                "description": source.description,
            }
            for source in db.execute(query).scalars().all()
        ]


def _metadata_fields(metadata: dict | None, domain_hint: str | None, topic_hint: str | None, description: str | None):
    metadata = metadata or {}
    topics = metadata.get("topics")
    if isinstance(topics, list):
        topic_hint = topic_hint or ", ".join(map(str, topics))
    topic_hint = topic_hint or (str(metadata.get("topic")) if metadata.get("topic") else None)
    domain_hint = domain_hint or (str(metadata.get("domain")) if metadata.get("domain") else None)
    description = description or (str(metadata.get("description")) if metadata.get("description") else None)
    return domain_hint, topic_hint, description


def _create_source(db, *, name, source_type, course_id, domain_hint, topic_hint, description, metadata_json, uri=None, local_path=None, content_hash=None):
    src = Source(
        name=name,
        source_type=source_type,
        uri=uri,
        local_path=local_path,
        course_id=course_id,
        domain_hint=domain_hint,
        topic_hint=topic_hint,
        description=description,
        metadata_json=metadata_json,
        content_hash=content_hash,
        status="queued",
    )
    db.add(src)
    db.flush()
    job = enqueue(db, src.id)
    return src, job


@app.post("/sources/file")
def upload_source(
    file: UploadFile = File(...), course_id: int | None = Form(None), domain_hint: str | None = Form(None),
    topic_hint: str | None = Form(None), description: str | None = Form(None), metadata_json: str | None = Form(None),
):
    path, digest = save_upload(file)
    metadata = None
    if metadata_json:
        try:
            metadata = parse_structured_literal(metadata_json)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    domain_hint, topic_hint, description = _metadata_fields(metadata, domain_hint, topic_hint, description)
    with session_scope() as db:
        existing = db.execute(select(Source).where(Source.content_hash == digest)).scalar_one_or_none()
        if existing:
            return {"source_id": existing.id, "job_id": None, "deduplicated": True}
        src, job = _create_source(
            db, name=file.filename, source_type="file", course_id=course_id,
            domain_hint=domain_hint, topic_hint=topic_hint, description=description,
            metadata_json=json.dumps(metadata, ensure_ascii=False) if metadata else metadata_json,
            local_path=str(path), content_hash=digest,
        )
        return {"source_id": src.id, "job_id": job.id, "status": "queued"}


@app.post("/sources/manual")
def add_manual_source(
    name: str = Form(...), payload: str = Form(...), input_format: str = Form("plain"),
    course_id: int | None = Form(None), domain_hint: str | None = Form(None), topic_hint: str | None = Form(None),
    description: str | None = Form(None),
):
    if not payload.strip():
        raise HTTPException(400, "Input is empty")
    metadata: dict = {}
    transcript = payload
    if input_format in {"json", "python_dict"}:
        try:
            obj = parse_structured_literal(payload)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        transcript = obj.get("transcript") or obj.get("text") or obj.get("content")
        if not isinstance(transcript, str) or not transcript.strip():
            raise HTTPException(400, "Structured input needs transcript, text, or content")
        metadata = {k: v for k, v in obj.items() if k not in {"transcript", "text", "content"}}
    elif input_format in {"srt", "vtt"}:
        transcript = clean_caption_text(payload)
    path, digest = save_text_content(f"{name}.txt", transcript)
    domain_hint, topic_hint, description = _metadata_fields(metadata, domain_hint, topic_hint, description)
    with session_scope() as db:
        existing = db.execute(select(Source).where(Source.content_hash == digest)).scalar_one_or_none()
        if existing:
            return {"source_id": existing.id, "job_id": None, "deduplicated": True}
        src, job = _create_source(
            db, name=name, source_type="transcript", course_id=course_id,
            domain_hint=domain_hint, topic_hint=topic_hint, description=description,
            metadata_json=json.dumps(metadata, ensure_ascii=False) if metadata else None,
            local_path=str(path), content_hash=digest,
        )
        return {"source_id": src.id, "job_id": job.id, "status": "queued", "parsed_format": input_format}


@app.post("/sources/transcript")
def add_manual_transcript(name: str = Form(...), transcript: str = Form(...), course_id: int | None = Form(None),
                          domain_hint: str | None = Form(None), topic_hint: str | None = Form(None), description: str | None = Form(None),
                          metadata_json: str | None = Form(None)):
    if not transcript.strip():
        raise HTTPException(400, "Transcript is empty")
    metadata = parse_structured_literal(metadata_json) if metadata_json else {}
    domain_hint, topic_hint, description = _metadata_fields(metadata, domain_hint, topic_hint, description)
    path, digest = save_text_content(f"{name}.txt", transcript)
    with session_scope() as db:
        existing = db.execute(select(Source).where(Source.content_hash == digest)).scalar_one_or_none()
        if existing:
            return {"source_id": existing.id, "job_id": None, "deduplicated": True}
        src, job = _create_source(
            db, name=name, source_type="transcript", course_id=course_id,
            domain_hint=domain_hint, topic_hint=topic_hint, description=description,
            metadata_json=json.dumps(metadata, ensure_ascii=False) if metadata else None,
            local_path=str(path), content_hash=digest,
        )
        return {"source_id": src.id, "job_id": job.id, "status": "queued"}


@app.post("/sources/structured")
def add_structured_source(req: StructuredSourceRequest):
    text = req.transcript or req.content
    if not text:
        raise HTTPException(400, "Provide transcript or content")
    path, digest = save_text_content(f"{req.name}.txt", text)
    with session_scope() as db:
        existing = db.execute(select(Source).where(Source.content_hash == digest)).scalar_one_or_none()
        if existing:
            return {"source_id": existing.id, "job_id": None, "deduplicated": True}
        metadata_json = json.dumps(req.metadata, ensure_ascii=False) if req.metadata else None
        domain_hint, topic_hint, description = _metadata_fields(req.metadata, req.domain_hint, req.topic_hint, req.description)
        src, job = _create_source(
            db, name=req.name, source_type="transcript", course_id=req.course_id,
            domain_hint=domain_hint, topic_hint=topic_hint, description=description,
            metadata_json=metadata_json, local_path=str(path), content_hash=digest,
        )
        return {"source_id": src.id, "job_id": job.id, "status": "queued"}


@app.post("/sources/youtube")
def add_youtube(url: HttpUrl, name: str | None = None, course_id: int | None = None,
                domain_hint: str | None = None, topic_hint: str | None = None, description: str | None = None,
                metadata_json: str | None = None):
    with session_scope() as db:
        src, job = _create_source(
            db, name=name or str(url), source_type="youtube", course_id=course_id,
            domain_hint=domain_hint, topic_hint=topic_hint, description=description,
            metadata_json=metadata_json, uri=str(url),
        )
        return {"source_id": src.id, "job_id": job.id, "status": "queued"}


@app.get("/jobs")
def jobs():
    with session_scope() as db:
        rows = db.execute(select(Job).order_by(Job.id.desc())).scalars().all()
        sources = {s.id: s for s in db.execute(select(Source)).scalars().all()}
        return [
            {
                "id": j.id,
                "source_id": j.source_id,
                "source_name": sources.get(j.source_id).name if sources.get(j.source_id) else None,
                "course_id": sources.get(j.source_id).course_id if sources.get(j.source_id) else None,
                "status": j.status,
                "stage": j.stage,
                "progress": f"{j.progress_current}/{j.progress_total}",
                "checkpoint": j.checkpoint,
                "plan": json.loads(j.plan_json) if j.plan_json else None,
                "error": j.error,
            }
            for j in rows
        ]


@app.get("/groups")
def groups():
    with session_scope() as db:
        return [
            {"id": g.id, "name": g.name, "description": g.description}
            for g in db.execute(select(Group).order_by(Group.name)).scalars().all()
        ]


@app.get("/courses/{course_id}/knowledge-graph")
def knowledge_graph(course_id: int):
    with session_scope() as db:
        concept_ids = {
            row[0]
            for row in db.execute(
                select(Evidence.concept_id)
                .join(Chunk, Evidence.chunk_id == Chunk.id)
                .join(Document, Chunk.document_id == Document.id)
                .join(Source, Document.source_id == Source.id)
                .where(Source.course_id == course_id)
            ).all()
        }
        if not concept_ids:
            return {"course_id": course_id, "concepts": [], "relations": []}

        concepts = db.execute(
            select(Concept).where(Concept.id.in_(concept_ids)).order_by(Concept.canonical_name)
        ).scalars().all()
        relations = db.execute(
            select(ConceptRelation).where(
                ConceptRelation.from_concept_id.in_(concept_ids),
                ConceptRelation.to_concept_id.in_(concept_ids),
            ).order_by(ConceptRelation.id)
        ).scalars().all()
        return {
            "course_id": course_id,
            "concepts": [
                {
                    "id": c.id,
                    "name": c.canonical_name,
                    "description": c.description,
                    "aliases": json.loads(c.aliases_json or "[]"),
                }
                for c in concepts
            ],
            "relations": [
                {
                    "from_concept_id": r.from_concept_id,
                    "to_concept_id": r.to_concept_id,
                    "type": r.relation_type,
                    "confidence": r.confidence,
                }
                for r in relations
            ],
        }


@app.get("/traces/{run_id}")
def traces(run_id: str):
    with session_scope() as db:
        return [
            {
                "id": t.id,
                "run_id": t.run_id,
                "event_type": t.event_type,
                "task": t.task,
                "model": t.model,
                "status": t.status,
                "latency_ms": t.latency_ms,
                "payload": json.loads(t.payload_json) if t.payload_json else None,
                "result": json.loads(t.result_json) if t.result_json else None,
                "error": t.error,
                "created_at": t.created_at.isoformat(),
            }
            for t in db.execute(
                select(AgentTrace).where(AgentTrace.run_id == run_id).order_by(AgentTrace.id)
            ).scalars().all()
        ]


@app.post("/study")
def study(req: StudyRequest):
    with session_scope() as db:
        return answer(db, provider(), req.question, req.group_id, req.top_k, req.course_id)


@app.get("/syllabus")
def syllabus(course_id: int | None = None):
    with session_scope() as db:
        return {"syllabus": build_syllabus(db, provider(), course_id)}


@app.post("/study-pack")
def study_pack(req: StudyPackRequest):
    with session_scope() as db:
        text_value, refs = build_study_pack_prompt(db, provider(), req.course_id, req.topic, req.top_k)
        return {"topic": req.topic, "study_text": text_value, "sources": refs, "copy_paste_ready": True}


@app.get("/courses/{course_id}/stats")
def course_stats(course_id: int):
    with session_scope() as db:
        sources = db.execute(select(Source).where(Source.course_id == course_id)).scalars().all()
        source_ids = {s.id for s in sources}
        docs = db.execute(
            select(Document).where(Document.source_id.in_(source_ids) if source_ids else False)
        ).scalars().all()
        doc_ids = {d.id for d in docs}
        chunks = db.execute(
            select(Chunk).where(Document.id.in_(doc_ids)).join(Document, Chunk.document_id == Document.id)
            if doc_ids
            else select(Chunk).where(False)
        ).scalars().all()
        leaf_chunks = [c for c in chunks if c.chunk_kind == "leaf"]
        parent_chunks = [c for c in chunks if c.chunk_kind == "parent"]
        concept_ids = {
            row[0]
            for row in db.execute(
                select(Evidence.concept_id)
                .join(Chunk, Evidence.chunk_id == Chunk.id)
                .join(Document, Chunk.document_id == Document.id)
                .join(Source, Document.source_id == Source.id)
                .where(Source.course_id == course_id)
            ).all()
        }
        relation_count = db.execute(
            select(ConceptRelation.id)
            .where(
                ConceptRelation.from_concept_id.in_(concept_ids),
                ConceptRelation.to_concept_id.in_(concept_ids),
            )
        ).scalars().all() if concept_ids else []
        return {
            "course_id": course_id,
            "sources": len(sources),
            "documents": len(docs),
            "chunks": len(leaf_chunks),
            "parent_chunks": len(parent_chunks),
            "embedded": sum(c.embedding_status == "done" for c in leaf_chunks),
            "grouped": sum(c.classification_status == "done" for c in leaf_chunks),
            "concept_indexed": sum(c.cluster_status == "done" for c in leaf_chunks),
            "concepts": len(concept_ids),
            "concept_relations": len(relation_count),
        }
