import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, Field
from sqlalchemy import select
from .config import settings
from .db import init_db, session_scope
from .models import Source, Job, Group, Course, Chunk, Document, Decision, SyllabusTopic, Syllabus
from .services.files import save_upload, save_text_content
from .services.queue import enqueue
from .services.worker import Worker
from .providers.base import DecisionPending
from .providers.factory import provider, ai_mode
from .services.study import answer
from .services.curriculum import generate_syllabus, get_syllabus, build_study_pack_prompt
from .services.decisions import record_decision, submit_answer, requeue_waiting_jobs, serialize
from .services.tutor import kickoff_message, next_message, part_message, parse_start_reply, set_topic_status, start_session, topic_overview
from .services.ingestion import parse_structured_literal, clean_caption_text

app = FastAPI(title=settings.app_name)
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


class AnswerRequest(BaseModel):
    answer: str


class SyllabusRequest(BaseModel):
    course_id: int | None = None


class TopicStatusRequest(BaseModel):
    status: str


class StartRequest(BaseModel):
    course_id: int | None = None
    start: int | None = None
    skip: list[int] = Field(default_factory=list)
    # The tutor's "START: 3; SKIP: 5, 7" line pasted back; overrides start/skip when present.
    reply: str | None = None


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
        "ai_mode": ai_mode(),
        "llm": bool(settings.gemini_api_key),
        "reasoning_model": settings.gemini_reasoning_model,
        "fast_model": settings.gemini_fast_model,
        "embedding_model": settings.gemini_embedding_model,
        "retrieval": "dense + BM25 + RRF + optional LLM rerank",
    }


@app.post("/courses")
def create_course(req: CourseRequest):
    with session_scope() as db:
        existing = db.execute(select(Course).where(Course.name == req.name)).scalar_one_or_none()
        if existing:
            return {"course_id": existing.id, "deduplicated": True}
        course = Course(name=req.name, description=req.description, domain=req.domain, outline=req.outline)
        db.add(course)
        db.flush()
        return {"course_id": course.id, "name": course.name}


@app.get("/courses")
def courses():
    with session_scope() as db:
        return [{"id": c.id, "name": c.name, "description": c.description, "domain": c.domain, "outline": c.outline}
                for c in db.execute(select(Course).order_by(Course.name)).scalars().all()]


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
    src = Source(name=name, source_type=source_type, uri=uri, local_path=local_path, course_id=course_id,
                 domain_hint=domain_hint, topic_hint=topic_hint, description=description,
                 metadata_json=metadata_json, content_hash=content_hash, status="queued")
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
        src, job = _create_source(db, name=file.filename, source_type="file", course_id=course_id,
                                  domain_hint=domain_hint, topic_hint=topic_hint, description=description,
                                  metadata_json=json.dumps(metadata, ensure_ascii=False) if metadata else metadata_json,
                                  local_path=str(path), content_hash=digest)
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
        src, job = _create_source(db, name=name, source_type="transcript", course_id=course_id,
                                  domain_hint=domain_hint, topic_hint=topic_hint, description=description,
                                  metadata_json=json.dumps(metadata, ensure_ascii=False) if metadata else None,
                                  local_path=str(path), content_hash=digest)
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
        src, job = _create_source(db, name=name, source_type="transcript", course_id=course_id,
                                  domain_hint=domain_hint, topic_hint=topic_hint, description=description,
                                  metadata_json=json.dumps(metadata, ensure_ascii=False) if metadata else None,
                                  local_path=str(path), content_hash=digest)
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
        src, job = _create_source(db, name=req.name, source_type="transcript", course_id=req.course_id,
                                  domain_hint=domain_hint, topic_hint=topic_hint, description=description,
                                  metadata_json=metadata_json, local_path=str(path), content_hash=digest)
        return {"source_id": src.id, "job_id": job.id, "status": "queued"}


@app.post("/sources/youtube")
def add_youtube(url: HttpUrl, name: str | None = None, course_id: int | None = None,
                domain_hint: str | None = None, topic_hint: str | None = None, description: str | None = None,
                metadata_json: str | None = None):
    with session_scope() as db:
        src, job = _create_source(db, name=name or str(url), source_type="youtube", course_id=course_id,
                                  domain_hint=domain_hint, topic_hint=topic_hint, description=description,
                                  metadata_json=metadata_json, uri=str(url))
        return {"source_id": src.id, "job_id": job.id, "status": "queued"}


@app.get("/jobs")
def jobs():
    with session_scope() as db:
        rows = db.execute(select(Job).order_by(Job.id.desc())).scalars().all()
        sources = {s.id: s for s in db.execute(select(Source)).scalars().all()}
        return [{"id": j.id, "source_id": j.source_id, "source_name": sources.get(j.source_id).name if sources.get(j.source_id) else None,
                 "course_id": sources.get(j.source_id).course_id if sources.get(j.source_id) else None,
                 "status": j.status, "stage": j.stage, "progress": f"{j.progress_current}/{j.progress_total}",
                 "checkpoint": j.checkpoint, "plan": json.loads(j.plan_json) if j.plan_json else None, "error": j.error} for j in rows]


@app.get("/groups")
def groups():
    with session_scope() as db:
        return [{"id": g.id, "name": g.name, "description": g.description} for g in db.execute(select(Group).order_by(Group.name)).scalars().all()]


@app.post("/study")
def study(req: StudyRequest):
    with session_scope() as db:
        return answer(db, provider(), req.question, req.group_id, req.top_k, req.course_id)


@app.get("/decisions")
def decisions(status: str | None = "pending"):
    with session_scope() as db:
        stmt = select(Decision).order_by(Decision.id)
        if status:
            stmt = stmt.where(Decision.status == status)
        return [serialize(d) for d in db.execute(stmt).scalars().all()]


@app.get("/decisions/{decision_id}")
def decision(decision_id: int):
    with session_scope() as db:
        d = db.get(Decision, decision_id)
        if d is None:
            raise HTTPException(404, "Decision not found")
        return {**serialize(d), "answer": d.answer}


@app.post("/decisions/{decision_id}/answer")
def answer_decision(decision_id: int, req: AnswerRequest):
    """Store the external AI's answer and resume whatever was waiting on it."""
    with session_scope() as db:
        try:
            d = submit_answer(db, decision_id, req.answer)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, f"{exc}. Paste the AI's full reply (a JSON object).") from exc
        requeued = requeue_waiting_jobs(db, d.id)
        payload = serialize(d, with_prompt=False)
        action, args = d.action, json.loads(d.action_args_json or "{}")
    follow_up = None
    if action == "syllabus":
        follow_up = _syllabus_response(args.get("course_id"))
        if isinstance(follow_up, JSONResponse):
            follow_up = json.loads(follow_up.body)
    return {"decision": payload, "requeued_jobs": requeued, "result": follow_up}


def _pending_response(pending: DecisionPending, **context) -> JSONResponse:
    with session_scope() as db:
        d = record_decision(db, pending, **context)
        body = {"status": "needs_decision", "decision": serialize(d),
                "message": "Copy the prompt into your external AI, then paste its reply via POST /decisions/{id}/answer."}
    return JSONResponse(status_code=202, content=body)


def _syllabus_payload(db, course_id: int | None) -> dict:
    s = get_syllabus(db, course_id)
    if s is None:
        return {"status": "missing", "syllabus": None, "topics": []}
    return {"status": "ready", "syllabus_id": s.id, "syllabus": s.text, "topics": topic_overview(db, s)}


def _syllabus_response(course_id: int | None):
    try:
        with session_scope() as db:
            s = generate_syllabus(db, provider(), course_id)
            if s is None:
                return {"status": "empty", "syllabus": None, "topics": [],
                        "message": "No organized learning material is available for this scope yet."}
            db.flush()
            return _syllabus_payload(db, course_id)
    except DecisionPending as pending:
        return _pending_response(pending, action="syllabus", action_args={"course_id": course_id})


@app.get("/syllabus")
def syllabus(course_id: int | None = None):
    with session_scope() as db:
        return _syllabus_payload(db, course_id)


@app.post("/syllabus")
def create_syllabus(req: SyllabusRequest):
    """(Re)generate and store the outline. Topics whose title is unchanged keep their status."""
    return _syllabus_response(req.course_id)


@app.post("/syllabus/topics/{topic_id}/status")
def topic_status(topic_id: int, req: TopicStatusRequest):
    with session_scope() as db:
        topic = db.get(SyllabusTopic, topic_id)
        if topic is None:
            raise HTTPException(404, "Topic not found")
        try:
            set_topic_status(db, topic, req.status)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"topic_id": topic.id, "status": topic.status}


def _require_syllabus(db, course_id: int | None):
    s = get_syllabus(db, course_id)
    if s is None:
        raise HTTPException(409, "Generate the study outline first (POST /syllabus).")
    return s


@app.get("/tutor/kickoff")
def tutor_kickoff(course_id: int | None = None):
    """First message for the external AI: the outline plus 'ask me where to start'."""
    with session_scope() as db:
        return {"text": kickoff_message(db, _require_syllabus(db, course_id))}


@app.post("/tutor/start")
def tutor_start(req: StartRequest):
    """Record where the user starts and what they skip, then return the first part to paste."""
    start, skip = (parse_start_reply(req.reply) if req.reply else (req.start, req.skip))
    if req.reply and start is None and not skip:
        raise HTTPException(400, "Could not find a 'START: n' or 'SKIP: ...' line in the reply.")
    with session_scope() as db:
        s = _require_syllabus(db, req.course_id)
        try:
            start_session(db, s, start, skip)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        db.flush()
        return {"start": start, "skip": skip, **next_message(db, s)}


@app.post("/tutor/next")
def tutor_next(req: SyllabusRequest):
    with session_scope() as db:
        return next_message(db, _require_syllabus(db, req.course_id))


@app.get("/tutor/topics/{topic_id}/parts/{part}")
def tutor_part(topic_id: int, part: int):
    """Re-send one part of a topic without changing progress."""
    with session_scope() as db:
        topic = db.get(SyllabusTopic, topic_id)
        if topic is None:
            raise HTTPException(404, "Topic not found")
        try:
            return part_message(db, db.get(Syllabus, topic.syllabus_id), topic_id, part)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc


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
        docs = db.execute(select(Document).where(Document.source_id.in_(source_ids) if source_ids else False)).scalars().all()
        doc_ids = {d.id for d in docs}
        chunks = db.execute(select(Chunk).where(Chunk.document_id.in_(doc_ids) if doc_ids else False)).scalars().all()
        return {
            "course_id": course_id,
            "sources": len(sources),
            "documents": len(docs),
            "chunks": len(chunks),
            "embedded": sum(c.embedding_status == "done" for c in chunks),
            "grouped": sum(c.classification_status == "done" for c in chunks),
            "concept_indexed": sum(c.cluster_status == "done" for c in chunks),
            "merged_duplicates": sum(c.duplicate_of_id is not None for c in chunks),
        }
