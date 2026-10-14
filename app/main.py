import json
from collections import defaultdict
from datetime import datetime, timedelta, UTC
from typing import Literal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
from sqlalchemy import select, or_, func, delete
from sqlalchemy.exc import IntegrityError
from .config import settings
from .db import init_db, session_scope
from .models import Source, Job, Group, Course, Chunk, Document, Concept, ConceptRelation, Evidence, AgentTrace, ChunkRelation, StudyProgress, StudyPlan, StudyPlanNode, Conversation, ConversationMessage, MaintenanceJob, SourceRelationship, Section, SourceSpace, TopicDetail
from .services.concepts import CONCEPT_PASS
from .services.files import save_upload, save_text_content
from .services.queue import enqueue
from .services.worker import Worker
from .providers.factory import provider
from .routers.learner import router as learner_router
from .services.study import answer
from .services.curriculum import build_study_pack_prompt, build_external_study_pack
from .services.plans import read_plan, request_updates_after_change, scope_key
from .services.units import SESSION_SIZES, OutlineTopic, StaleTopic, TopicNotFound, prepare_topic, topic_lock
from .services.handoff import NoPlan, master_prompt
from .services.profile import get_profile, profile_json, save_profile
from .services.arrangement import ArrangementError, NoTopics, apply_reply, apply_start_and_skip, clear as clear_arrangement, done_sessions, \
    organize_request, session_payload, sessions as plan_sessions, sync_progress
from .services.progress import current_position, plan_topic_of, remember_position, roll_up, save_progress, set_skipped, skipped_ids
from .services.ingestion import parse_structured_literal, clean_caption_text, canonical_youtube_url, chunk_text
from .services.spaces import also_in, in_space, spaces_of
from .services.organize import index_pending, queue_organization
from .services.cleanup import delete_files, delete_sources, remove_empty_spaces, remove_space, request_compaction
from .services.placement import keep_here, keep_out, request_placement_check, suggestions as placement_suggestions

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(learner_router)
worker: Worker | None = None


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=3000)


class StudyRequest(BaseModel):
    question: str
    group_id: int | None = None
    course_id: int | None = None
    top_k: int = Field(default=8, ge=1, le=50)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)
    retrieval_scope: Literal["space_only", "space_and_related", "all_sources"] = "space_only"


class StudyPackRequest(BaseModel):
    topic: str
    course_id: int | None = None
    top_k: int = Field(default=12, ge=1, le=30)
    topic_id: str | None = None
    retrieval_scope: Literal["space_only", "space_and_related", "all_sources"] = "space_only"


class ExternalStudyPackRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=500)
    course_id: int | None = None
    top_k: int = Field(default=12, ge=1, le=20)
    max_input_tokens: int = Field(default=6000, ge=1000, le=24000)
    tutor: Literal["general", "chatgpt", "gemini", "claude"] = "general"
    topic_id: str | None = None
    retrieval_scope: Literal["space_only", "space_and_related", "all_sources"] = "space_only"


class PrepareTopicRequest(BaseModel):
    node_id: str = Field(min_length=1, max_length=100)
    course_id: int | None = None
    size: Literal["compact", "standard", "large", "xl"] = "standard"
    tutor: Literal["general", "chatgpt", "gemini", "claude"] = "general"
    # False when the topic is only shown again (the tutor changed, back to a parent): a result made
    # while the AI was unavailable is then returned as it is instead of being prepared again.
    retry: bool = True


class StudyProgressRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=500)
    topic_id: str | None = Field(default=None, max_length=100)
    course_id: int | None = None
    completed: bool = True


class CourseRequest(BaseModel):
    name: str
    description: str | None = None
    domain: str | None = None
    outline: str | None = None


class SourceCourseRequest(BaseModel):
    course_id: int | None = None


class StructuredSourceRequest(BaseModel):
    name: str
    course_id: int | None = None
    assignment_mode: Literal["auto", "explicit_course", "explicit_unassigned"] = "explicit_unassigned"
    domain_hint: str | None = None
    topic_hint: str | None = None
    description: str | None = None
    metadata: dict | None = None
    transcript: str | None = None
    content: str | None = None


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="New chat", max_length=500)


class TextContentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000_000)
    input_format: Literal["plain", "srt", "vtt"] = "plain"


class ConversationMessageRequest(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)
    metadata: dict | None = None


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


@app.get("/conversations")
def list_conversations():
    with session_scope() as db:
        return [{"id": c.id, "title": c.title, "created_at": c.created_at.isoformat(), "updated_at": c.updated_at.isoformat()} for c in db.execute(select(Conversation).order_by(Conversation.updated_at.desc())).scalars().all()]


@app.post("/conversations")
def create_conversation(req: ConversationCreateRequest):
    from uuid import uuid4
    with session_scope() as db:
        row = Conversation(id=str(uuid4()), title=req.title.strip() or "New chat")
        db.add(row)
        db.flush()
        return {"id": row.id, "title": row.title}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    with session_scope() as db:
        row = db.get(Conversation, conversation_id)
        if row is None:
            raise HTTPException(404, "Conversation not found")
        messages = db.execute(select(ConversationMessage).where(ConversationMessage.conversation_id == conversation_id).order_by(ConversationMessage.id)).scalars().all()
        return {"id": row.id, "title": row.title, "messages": [{"id": str(m.id), "role": m.role, "content": m.content, "metadata": json.loads(m.metadata_json or "{}")} for m in messages]}


@app.post("/conversations/{conversation_id}/messages")
def add_conversation_message(conversation_id: str, req: ConversationMessageRequest):
    with session_scope() as db:
        row = db.get(Conversation, conversation_id)
        if row is None:
            raise HTTPException(404, "Conversation not found")
        message = ConversationMessage(conversation_id=conversation_id, role=req.role, content=req.content, metadata_json=json.dumps(req.metadata or {}, ensure_ascii=False))
        db.add(message)
        if req.role == "user" and row.title == "New chat":
            row.title = req.content.strip().splitlines()[0][:90] or "New chat"
        row.updated_at = datetime.now(UTC)
        db.flush()
        return {"id": str(message.id), "role": message.role, "content": message.content}


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    with session_scope() as db:
        row = db.get(Conversation, conversation_id)
        if row is None:
            raise HTTPException(404, "Conversation not found")
        db.execute(delete(ConversationMessage).where(ConversationMessage.conversation_id == conversation_id))
        db.delete(row)
        return {"deleted": True, "id": conversation_id}


@app.post("/courses")
def create_course(req: CourseRequest):
    with session_scope() as db:
        existing = db.execute(select(Course).where(Course.name == req.name)).scalar_one_or_none()
        if existing:
            return {"id": existing.id, "name": existing.name, "deduplicated": True}
        course = Course(name=req.name, description=req.description, domain=req.domain, outline=req.outline, created_by="you")
        db.add(course)
        db.flush()
        return {"id": course.id, "name": course.name}


@app.get("/courses")
def courses():
    with session_scope() as db:
        filed = dict(db.execute(select(Source.course_id, func.count(Source.id)).where(Source.course_id.is_not(None)).group_by(Source.course_id)).all())
        shown = dict(db.execute(select(SourceSpace.course_id, func.count(SourceSpace.id)).group_by(SourceSpace.course_id)).all())
        return [
            {"id": c.id, "name": c.name, "description": c.description, "domain": c.domain, "outline": c.outline,
             "created_by": c.created_by, "source_count": filed.get(c.id, 0) + shown.get(c.id, 0)}
            for c in db.execute(select(Course).order_by(Course.name)).scalars().all()
        ]


@app.delete("/courses/{course_id}")
def delete_course(course_id: int, with_material: bool = False):
    """Delete a study space. Its material is kept in All sources, or deleted with it when asked."""
    local_paths: list = []
    with session_scope() as db:
        course = db.get(Course, course_id)
        if course is None:
            raise HTTPException(status_code=404, detail="Study space not found")
        filed = db.execute(select(Source).where(Source.course_id == course_id)).scalars().all()
        if with_material and filed:
            busy = db.execute(select(Job.id).where(Job.source_id.in_([s.id for s in filed]), Job.status.in_(["queued", "running", "retrying"])).limit(1)).first()
            if busy:
                raise HTTPException(status_code=409, detail="Some material in this space is still being processed. Try again when it is ready.")
        material_changed = with_material and bool(filed)
        local_paths = remove_space(db, course, with_material=with_material)
        if material_changed:
            request_updates_after_change(db, set())
        request_compaction(db)
    delete_files(local_paths)
    return {"deleted": True, "course_id": course_id, "removed_sources": len(filed) if with_material else 0}


@app.get("/sources")
def sources(course_id: int | None = None):
    with session_scope() as db:
        query = select(Source).order_by(Source.created_at.desc())
        if course_id is not None:
            query = query.where(in_space(course_id))
        source_rows = db.execute(query).scalars().all()
        # Older versions could create multiple rows for the same YouTube URL.
        # Keep the most useful record visible, preferring completed ingestion.
        unique_sources: dict[tuple[str, str | int], Source] = {}
        failed_attempts: dict[tuple[str, str | int], int] = defaultdict(int)
        source_keys: dict[int, tuple[str, str | int]] = {}
        for source in source_rows:
            if source.source_type == "youtube":
                key = ("youtube", canonical_youtube_url(source.uri or source.name))
            else:
                key = ("source", source.id)
            source_keys[source.id] = key
            if source.status == "failed":
                failed_attempts[key] += 1
            current = unique_sources.get(key)
            if current is None or (source.status == "done" and current.status != "done"):
                unique_sources[key] = source
        visible = list(unique_sources.values())
        ids = [source.id for source in visible]
        course_names = {course.id: course.name for course in db.execute(select(Course)).scalars().all()}
        document_counts = dict(db.execute(
            select(Document.source_id, func.count(Document.id)).where(Document.source_id.in_(ids)).group_by(Document.source_id)
        ).all()) if ids else {}
        section_counts = dict(db.execute(
            select(Section.source_id, func.count(Section.id)).where(Section.source_id.in_(ids)).group_by(Section.source_id)
        ).all()) if ids else {}
        # What organizing still has to do: leaves without a section, and leaves not yet read
        # in full for concepts (only when concept extraction is enabled).
        def leaf_counts(condition):
            return dict(db.execute(
                select(Document.source_id, func.count(Chunk.id)).join(Chunk, Chunk.document_id == Document.id)
                .where(Document.source_id.in_(ids), Chunk.chunk_kind == "leaf", condition)
                .group_by(Document.source_id)
            ).all()) if ids else {}
        pending_counts = leaf_counts(index_pending())
        sections_pending = leaf_counts(Chunk.section_id.is_(None))
        concepts_pending = leaf_counts(Chunk.concept_pass < CONCEPT_PASS) if settings.concept_extraction else {}
        concept_counts = dict(db.execute(
            select(Document.source_id, func.count(func.distinct(Evidence.concept_id)))
            .select_from(Evidence).join(Chunk, Evidence.chunk_id == Chunk.id).join(Document, Chunk.document_id == Document.id)
            .where(Document.source_id.in_(ids)).group_by(Document.source_id)
        ).all()) if ids else {}
        shown_also = also_in(db, ids)
        placements = placement_suggestions(db, visible)
        # Parts of a source that could not be read, e.g. playlist videos without captions.
        issues: dict[int, list[dict]] = defaultdict(list)
        if ids:
            for document in db.execute(
                select(Document).where(Document.source_id.in_(ids), Document.status.in_(["needs_input", "failed"])).order_by(Document.id)
            ).scalars().all():
                issues[document.source_id].append({"document_id": document.id, "title": document.title or document.external_id,
                                                   "status": document.status, "message": document.error})
        return [
            {
                "id": source.id,
                "name": source.name,
                "source_type": source.source_type,
                "course_id": source.course_id,
                "course_name": course_names.get(source.course_id),
                "assignment_mode": source.assignment_mode,
                "assignment_status": source.assignment_status,
                "assignment_reason": source.assignment_reason,
                "status": source.status,
                # Only the user-facing needs-input message; raw failure details stay server-side.
                "status_message": source.error if source.status == "needs_input" else None,
                "topic_hint": source.topic_hint,
                "domain_hint": source.domain_hint,
                "description": source.description,
                "failed_attempt_count": failed_attempts[source_keys[source.id]],
                "document_count": document_counts.get(source.id, 0),
                "section_count": section_counts.get(source.id, 0),
                "concept_count": concept_counts.get(source.id, 0),
                "index_pending": pending_counts.get(source.id, 0),
                "sections_pending": sections_pending.get(source.id, 0),
                "concepts_pending": concepts_pending.get(source.id, 0),
                "issues": issues.get(source.id, []),
                # Spaces it is also shown in, besides the one it is filed in.
                "also_in": [{"course_id": c, "course_name": course_names.get(c)} for c in shown_also.get(source.id, [])],
                # What its content suggests about where it belongs; the learner decides.
                "placement": placements.get(source.id),
            }
            for source in visible
        ]


@app.patch("/sources/{source_id}/course")
def move_source_to_course(source_id: int, req: SourceCourseRequest):
    with session_scope() as db:
        source = db.get(Source, source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        if req.course_id is not None and db.get(Course, req.course_id) is None:
            raise HTTPException(status_code=404, detail="Course not found")
        _file_source(db, source, req.course_id, "Changed explicitly in the library.")
        return {"id": source.id, "course_id": source.course_id}


def _assignment_conflict(db, existing: Source, assignment_mode: str, course_id: int | None) -> bool:
    """A source added again conflicts only when it was asked to be in a space it is not in (filed or shown),
    or to be unfiled while it is filed."""
    if assignment_mode == "explicit_course":
        return course_id not in spaces_of(db, [existing.id])
    return assignment_mode == "explicit_unassigned" and existing.course_id is not None


def _file_source(db, source: Source, course_id: int | None, reason: str) -> None:
    """File a source in a space (its primary place). A space it was only shown in becomes its home."""
    previous_course = source.course_id
    source.course_id = course_id
    source.assignment_mode = "explicit_course" if course_id is not None else "explicit_unassigned"
    source.assignment_status = "assigned" if course_id is not None else "unassigned"
    source.assignment_reason = reason
    # A space it was only shown in becomes its home; an unfiled source is shown in no space at all.
    links = select(SourceSpace.course_id).where(SourceSpace.source_id == source.id)
    if course_id is not None:
        links = links.where(SourceSpace.course_id == course_id)
    dropped = set(db.execute(links).scalars().all())
    db.execute(delete(SourceSpace).where(SourceSpace.source_id == source.id, SourceSpace.course_id.in_(dropped)))
    db.flush()
    if source.status == "done" and (previous_course != course_id or dropped):
        request_updates_after_change(db, {previous_course, course_id} | dropped)
    if source.status == "done" and previous_course != course_id:
        request_placement_check(db, [source.id])  # how it fits its new neighbours is judged in the background
    if previous_course != course_id:
        remove_empty_spaces(db, {previous_course})  # a space LearnOS made is removed once nothing is left in it


class PlacementRequest(BaseModel):
    action: Literal["move", "show_in_both", "keep"]
    course_id: int


@app.post("/sources/{source_id}/placement")
def answer_placement(source_id: int, req: PlacementRequest):
    """The learner's answer to a placement suggestion: move the source, show it in both spaces, or keep it."""
    with session_scope() as db:
        source = db.get(Source, source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        course = db.get(Course, req.course_id)
        if course is None:
            raise HTTPException(status_code=404, detail="Course not found")
        if req.action == "move":
            previous = source.course_id
            _file_source(db, source, course.id, f'Moved to "{course.name}" because its content matches sources there.')
            if previous is not None and previous != course.id:
                keep_out(source, previous)  # moved rather than shown in both: do not offer the old space back
        elif req.action == "show_in_both":
            if source.course_id is None:
                raise HTTPException(status_code=422, detail="File this source in a space first; it can then also be shown in others")
            if course.id != source.course_id and db.execute(select(SourceSpace.id).where(
                    SourceSpace.source_id == source.id, SourceSpace.course_id == course.id)).first() is None:
                db.add(SourceSpace(source_id=source.id, course_id=course.id))
                db.flush()
                if source.status == "done":
                    request_updates_after_change(db, {course.id})
        else:
            keep_here(source, course.id)
        return {"id": source.id, "course_id": source.course_id,
                "also_in": [{"course_id": c, "course_name": db.get(Course, c).name} for c in also_in(db, [source.id]).get(source.id, [])]}


@app.delete("/sources/{source_id}/spaces/{course_id}")
def stop_showing_source(source_id: int, course_id: int):
    """Stop showing a source in a space it is only shown in; where it is filed does not change."""
    with session_scope() as db:
        link = db.execute(select(SourceSpace).where(SourceSpace.source_id == source_id, SourceSpace.course_id == course_id)).scalar_one_or_none()
        if link is None:
            raise HTTPException(status_code=404, detail="This source is not shown in that space")
        source = db.get(Source, source_id)
        db.delete(link)
        if source is not None:
            keep_out(source, course_id)  # and do not suggest it there again
        db.flush()
        if source is not None and source.status == "done":
            request_updates_after_change(db, {course_id})
        return {"id": source_id, "removed_from": course_id}


@app.delete("/sources/{source_id}")
def remove_source(source_id: int, failed_attempts_only: bool = False):
    from pathlib import Path

    local_paths: list[Path] = []
    with session_scope() as db:
        source = db.get(Source, source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        related_sources = [source]
        if source.source_type == "youtube":
            canonical = canonical_youtube_url(source.uri or source.name)
            related_sources = [item for item in db.execute(
                select(Source).where(Source.source_type == "youtube")
            ).scalars().all() if canonical_youtube_url(item.uri or item.name) == canonical]
        if failed_attempts_only:
            related_sources = [item for item in related_sources if item.status == "failed"]
            if not related_sources:
                return {"removed": True, "source_id": source_id, "removed_count": 0}
        source_ids = {item.id for item in related_sources}
        active_job = db.execute(
            select(Job.id).where(Job.source_id.in_(source_ids), Job.status.in_(["queued", "running", "retrying"])).limit(1)
        ).first()
        if active_job:
            raise HTTPException(status_code=409, detail="This source is still processing. Wait until it finishes before removing it.")
        # Organizing reads sources without a Job; removing the one being read right now would leave links
        # to deleted concepts. Sources still waiting in a run are fine: the worker skips removed sources.
        heartbeat_cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=settings.maintenance_stale_seconds)
        for maintenance in db.execute(select(MaintenanceJob).where(MaintenanceJob.status.in_(["queued", "running"]))).scalars().all():
            live = worker is not None and worker.current_maintenance_id == maintenance.id
            if maintenance.status == "running" and not live and maintenance.updated_at and maintenance.updated_at < heartbeat_cutoff:
                continue  # left over from an interrupted run; the worker will requeue it
            if maintenance.status != "running":
                continue
            pending = json.loads(maintenance.items_json or "[]")[maintenance.progress_current:maintenance.progress_current + 1]
            if any(isinstance(item, dict) and item.get("kind") in {"concepts", "source"} and int(item.get("id", -1)) in source_ids for item in pending):
                raise HTTPException(status_code=409, detail="LearnOS is still organizing this source in the background. Try again in a moment, when LearnOS has finished organizing it.")

        files, affected_courses = delete_sources(db, related_sources)
        local_paths.extend(files)
        removed_spaces = remove_empty_spaces(db, affected_courses)
        affected_courses -= set(removed_spaces)
        request_compaction(db)
        if affected_courses:
            request_updates_after_change(db, affected_courses)

    delete_files(local_paths)
    return {"removed": True, "source_id": source_id, "removed_count": len(source_ids), "removed_spaces": removed_spaces}


def _has_active_job(db, source_id: int) -> bool:
    return db.execute(
        select(Job.id).where(Job.source_id == source_id, Job.status.in_(["queued", "running", "retrying"])).limit(1)
    ).first() is not None


def _pasted_text(req: TextContentRequest) -> str:
    text = clean_caption_text(req.text) if req.input_format in {"srt", "vtt"} else req.text
    if not text.strip():
        raise HTTPException(400, "The pasted text is empty.")
    return text


@app.post("/documents/{document_id}/transcript")
def add_document_transcript(document_id: int, req: TextContentRequest):
    """Attach a pasted transcript to one item (e.g. a video without captions) and index it."""
    pieces = chunk_text(_pasted_text(req), settings.chunk_size, settings.chunk_overlap)
    with session_scope() as db:
        document = db.get(Document, document_id)
        if document is None:
            raise HTTPException(404, "Item not found")
        source = db.get(Source, document.source_id)
        if _has_active_job(db, source.id):
            raise HTTPException(409, "This source is still processing. Try again when it finishes.")
        if db.execute(select(Chunk.id).where(Chunk.document_id == document.id, Chunk.chunk_kind == "leaf").limit(1)).first():
            raise HTTPException(409, "This item already has text.")
        for index, piece in enumerate(pieces):
            db.add(Chunk(document_id=document.id, chunk_index=index, chunk_kind="leaf", text=piece, token_estimate=max(1, len(piece) // 4)))
        try:
            metadata = json.loads(document.metadata_json or "{}")
        except json.JSONDecodeError:
            metadata = {}
        metadata = metadata if isinstance(metadata, dict) else {}
        metadata["transcript_origin"] = "pasted"
        document.metadata_json = json.dumps(metadata, ensure_ascii=False)
        document.status = "processing"
        document.error = None
        source.status = "queued"
        source.error = None
        job = enqueue(db, source.id)
        return {"document_id": document.id, "source_id": source.id, "job_id": job.id, "chunks": len(pieces)}


@app.post("/sources/{source_id}/text")
def replace_source_text(source_id: int, req: TextContentRequest):
    """Give a file source with no readable text (e.g. a scanned PDF) its text, then index it."""
    from pathlib import Path

    text = _pasted_text(req)
    with session_scope() as db:
        source = db.get(Source, source_id)
        if source is None:
            raise HTTPException(404, "Source not found")
        if source.source_type == "youtube":
            raise HTTPException(409, "For YouTube sources, paste a transcript for each video instead.")
        if _has_active_job(db, source.id):
            raise HTTPException(409, "This source is still processing. Try again when it finishes.")
        has_text = db.execute(
            select(Chunk.id).join(Document, Chunk.document_id == Document.id)
            .where(Document.source_id == source.id, Chunk.chunk_kind == "leaf").limit(1)
        ).first()
        if has_text:
            raise HTTPException(409, "This source already has readable text. Remove it and add the new version instead.")
        new_path, digest = save_text_content(f"{Path(source.name).stem or 'source'}.txt", text)
        duplicate = db.execute(select(Source).where(Source.content_hash == digest, Source.id != source.id)).scalar_one_or_none()
        if duplicate:
            new_path.unlink(missing_ok=True)
            raise HTTPException(409, f'This text is already in your library as "{duplicate.name}".')
        old_path = source.local_path
        db.execute(delete(Document).where(Document.source_id == source.id))
        source.local_path = str(new_path)
        source.content_hash = digest
        source.status = "queued"
        source.error = None
        job = enqueue(db, source.id)
        result = {"source_id": source.id, "job_id": job.id}
    if old_path and old_path != str(new_path):
        candidate = Path(old_path).resolve()
        if candidate.is_relative_to(settings.upload_path.resolve()) and candidate.exists():
            candidate.unlink()
    return result


@app.post("/sources/{source_id}/reprocess")
def reprocess_source(source_id: int):
    """Try the unreadable parts of a source again, keeping everything already processed."""
    with session_scope() as db:
        source = db.get(Source, source_id)
        if source is None:
            raise HTTPException(404, "Source not found")
        if _has_active_job(db, source.id):
            raise HTTPException(409, "This source is already processing.")
        retried = 0
        for document in db.execute(
            select(Document).where(Document.source_id == source.id, Document.status.in_(["needs_input", "failed"]))
        ).scalars().all():
            document.status = "pending"
            document.error = None
            retried += 1
        source.status = "queued"
        source.error = None
        job = enqueue(db, source.id)
        return {"source_id": source.id, "job_id": job.id, "documents_retried": retried}


def _metadata_fields(metadata: dict | None, domain_hint: str | None, topic_hint: str | None, description: str | None):
    metadata = metadata or {}
    topics = metadata.get("topics")
    if isinstance(topics, list):
        topic_hint = topic_hint or ", ".join(map(str, topics))
    topic_hint = topic_hint or (str(metadata.get("topic")) if metadata.get("topic") else None)
    domain_hint = domain_hint or (str(metadata.get("domain")) if metadata.get("domain") else None)
    description = description or (str(metadata.get("description")) if metadata.get("description") else None)
    return domain_hint, topic_hint, description


def _create_source(db, *, name, source_type, course_id, assignment_mode, domain_hint, topic_hint, description, metadata_json, uri=None, local_path=None, content_hash=None):
    if assignment_mode not in {"auto", "explicit_course", "explicit_unassigned"}:
        raise HTTPException(422, "Invalid source assignment mode")
    if assignment_mode == "explicit_course" and course_id is None:
        raise HTTPException(422, "Choose a learning space for explicit_course assignment")
    if assignment_mode == "explicit_course" and db.get(Course, course_id) is None:
        raise HTTPException(404, "Course not found")
    if assignment_mode != "explicit_course":
        course_id = None
    src = Source(
        name=name,
        source_type=source_type,
        uri=uri,
        local_path=local_path,
        course_id=course_id,
        assignment_mode=assignment_mode,
        assignment_status="pending" if assignment_mode == "auto" else "assigned" if course_id is not None else "unassigned",
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
    file: UploadFile = File(...), course_id: int | None = Form(None), assignment_mode: str = Form("explicit_unassigned"), domain_hint: str | None = Form(None),
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
            conflict = _assignment_conflict(db, existing, assignment_mode, course_id)
            return {"source_id": existing.id, "job_id": None, "deduplicated": True, "assignment_preserved": True, "assignment_conflict": conflict, "course_id": existing.course_id}
        src, job = _create_source(
            db, name=file.filename, source_type="file", course_id=course_id, assignment_mode=assignment_mode,
            domain_hint=domain_hint, topic_hint=topic_hint, description=description,
            metadata_json=json.dumps(metadata, ensure_ascii=False) if metadata else metadata_json,
            local_path=str(path), content_hash=digest,
        )
        return {"source_id": src.id, "job_id": job.id, "status": "queued"}


@app.post("/sources/manual")
def add_manual_source(
    name: str = Form(...), payload: str = Form(...), input_format: str = Form("plain"),
    course_id: int | None = Form(None), assignment_mode: str = Form("explicit_unassigned"), domain_hint: str | None = Form(None), topic_hint: str | None = Form(None),
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
    extension = input_format if input_format in {"srt", "vtt"} else "txt"
    path, digest = save_text_content(f"{name}.{extension}", transcript)
    domain_hint, topic_hint, description = _metadata_fields(metadata, domain_hint, topic_hint, description)
    with session_scope() as db:
        existing = db.execute(select(Source).where(Source.content_hash == digest)).scalar_one_or_none()
        if existing:
            conflict = _assignment_conflict(db, existing, assignment_mode, course_id)
            return {"source_id": existing.id, "job_id": None, "deduplicated": True, "assignment_preserved": True, "assignment_conflict": conflict, "course_id": existing.course_id}
        src, job = _create_source(
            db, name=name, source_type="transcript", course_id=course_id, assignment_mode=assignment_mode,
            domain_hint=domain_hint, topic_hint=topic_hint, description=description,
            metadata_json=json.dumps(metadata, ensure_ascii=False) if metadata else None,
            local_path=str(path), content_hash=digest,
        )
        return {"source_id": src.id, "job_id": job.id, "status": "queued", "parsed_format": input_format}


@app.post("/sources/transcript")
def add_manual_transcript(name: str = Form(...), transcript: str = Form(...), course_id: int | None = Form(None), assignment_mode: str = Form("explicit_unassigned"),
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
            conflict = _assignment_conflict(db, existing, assignment_mode, course_id)
            return {"source_id": existing.id, "job_id": None, "deduplicated": True, "assignment_preserved": True, "assignment_conflict": conflict, "course_id": existing.course_id}
        src, job = _create_source(
            db, name=name, source_type="transcript", course_id=course_id, assignment_mode=assignment_mode,
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
            conflict = _assignment_conflict(db, existing, req.assignment_mode, req.course_id)
            return {"source_id": existing.id, "job_id": None, "deduplicated": True, "assignment_preserved": True, "assignment_conflict": conflict, "course_id": existing.course_id}
        metadata_json = json.dumps(req.metadata, ensure_ascii=False) if req.metadata else None
        domain_hint, topic_hint, description = _metadata_fields(req.metadata, req.domain_hint, req.topic_hint, req.description)
        src, job = _create_source(
            db, name=req.name, source_type="transcript", course_id=req.course_id, assignment_mode=req.assignment_mode,
            domain_hint=domain_hint, topic_hint=topic_hint, description=description,
            metadata_json=metadata_json, local_path=str(path), content_hash=digest,
        )
        return {"source_id": src.id, "job_id": job.id, "status": "queued"}


@app.post("/sources/youtube")
def add_youtube(url: HttpUrl, name: str | None = None, course_id: int | None = None, assignment_mode: Literal["auto", "explicit_course", "explicit_unassigned"] = "explicit_unassigned",
                domain_hint: str | None = None, topic_hint: str | None = None, description: str | None = None,
                metadata_json: str | None = None):
    with session_scope() as db:
        canonical_url = canonical_youtube_url(str(url))
        existing_sources = db.execute(
            select(Source).where(Source.source_type == "youtube").order_by(Source.id.desc())
        ).scalars().all()
        existing = next((s for s in existing_sources if canonical_youtube_url(s.uri or "") == canonical_url), None)
        if existing:
            existing_job = db.execute(
                select(Job).where(Job.source_id == existing.id).order_by(Job.id.desc()).limit(1)
            ).scalar_one_or_none()
            return {
                "source_id": existing.id,
                "job_id": existing_job.id if existing_job else None,
                "status": existing_job.status if existing_job else existing.status,
                "deduplicated": True,
                "assignment_preserved": True,
                "course_id": existing.course_id,
                "assignment_conflict": _assignment_conflict(db, existing, assignment_mode, course_id),
            }
        src, job = _create_source(
            db, name=name or str(url), source_type="youtube", course_id=course_id, assignment_mode=assignment_mode,
            domain_hint=domain_hint, topic_hint=topic_hint, description=description,
            metadata_json=metadata_json, uri=canonical_url,
        )
        return {"source_id": src.id, "job_id": job.id, "status": "queued"}


@app.post("/jobs/{job_id}/retry")
def retry_job(job_id: int):
    with session_scope() as db:
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status != "failed":
            raise HTTPException(409, "Only failed jobs can be retried")
        src = db.get(Source, job.source_id)
        if not src:
            raise HTTPException(409, "Job source no longer exists")
        # Reset the automatic retry budget for this explicit resume while retaining
        # checkpoints and completed embeddings/transcripts.
        job.status = "queued"
        job.retry_count = 0
        job.error = None
        job.error_message = None
        job.updated_at = datetime.now(UTC)
        src.status = "queued"
        src.error = None
        return {"job_id": job.id, "source_id": src.id, "status": job.status, "checkpoint": job.checkpoint}


@app.get("/jobs")
def jobs():
    with session_scope() as db:
        rows = db.execute(select(Job).order_by(Job.id.desc())).scalars().all()
        sources = {s.id: s for s in db.execute(select(Source)).scalars().all()}
        embedded_by_source = dict(db.execute(
            select(Document.source_id, func.count(Chunk.id))
            .join(Chunk, Chunk.document_id == Document.id)
            .where(Chunk.chunk_kind == "leaf", Chunk.embedding_status == "done")
            .group_by(Document.source_id)
        ).all())
        return [
            {
                "id": j.id,
                "source_id": j.source_id,
                "source_name": sources.get(j.source_id).name if sources.get(j.source_id) else None,
                "source_type": sources.get(j.source_id).source_type if sources.get(j.source_id) else None,
                "source_uri": sources.get(j.source_id).uri if sources.get(j.source_id) else None,
                "course_id": sources.get(j.source_id).course_id if sources.get(j.source_id) else None,
                "status": j.status,
                "stage": j.stage,
                "progress": f"{embedded_by_source.get(j.source_id, 0) if j.stage == 'index' else j.progress_current}/{j.progress_total}",
                "checkpoint": j.checkpoint,
                "retry_count": j.retry_count,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "started_at": j.started_at.isoformat() if j.started_at else None,
                "updated_at": j.updated_at.isoformat() if j.updated_at else None,
                "plan": json.loads(j.plan_json) if j.plan_json else None,
                "error_message": _job_error_message(j),
            }
            for j in rows
        ]


def _job_error_message(job: Job) -> str | None:
    detail = (job.error or "").lower()
    if "429" in detail or "too many requests" in detail:
        phase = "embedding" if "batchembedcontents" in detail or job.stage == "index" else job.stage.replace("_", " ")
        if any(marker in detail for marker in ("perday", "per_day", "per day", "daily quota", "quota_exceeded", "daily limit")):
            return f"Gemini reports its daily quota for {phase} is exhausted or unavailable. Automatic retries stopped. Daily quotas reset at midnight Pacific; if the limit is unavailable or zero, check the Gemini project/model quota. Choose Resume after recovery, or switch provider/model."
        if job.status == "failed":
            return f"Gemini rate limit or quota persisted during {phase}; automatic retries stopped after {settings.retry_max} attempts. Completed work is saved. Check Gemini usage, then choose Resume or switch provider/model."
        return f"Gemini temporary rate limit during {phase}. LearnOS is retrying automatically ({job.retry_count}/{settings.retry_max})."
    return job.error_message


@app.post("/maintenance/organize-existing")
def enqueue_existing_organization(include_unassigned: bool = True):
    with session_scope() as db:
        row, running = queue_organization(db, include_unassigned=include_unassigned)
        return {"id": row.id, "status": row.status, "progress_current": row.progress_current, "progress_total": row.progress_total,
                "checkpoint": row.checkpoint, "deduplicated": running}


@app.get("/maintenance/jobs")
def maintenance_jobs(task: str | None = None, active: bool = False):
    """Background jobs, newest first; optionally one task, or only those waiting or running."""
    with session_scope() as db:
        query = select(MaintenanceJob).order_by(MaintenanceJob.id.desc())
        if task:
            query = query.where(MaintenanceJob.task == task)
        if active:
            query = query.where(MaintenanceJob.status.in_(["queued", "running"]))
        rows = db.execute(query).scalars().all()
        return [{"id": row.id, "task": row.task, "status": row.status, "progress_current": row.progress_current, "progress_total": row.progress_total, "checkpoint": row.checkpoint, "errors": json.loads(row.errors_json or "{}"), "updated_at": row.updated_at.isoformat() if row.updated_at else None} for row in rows]


@app.post("/maintenance/jobs/{job_id}/retry")
def retry_maintenance_job(job_id: int):
    with session_scope() as db:
        row = db.get(MaintenanceJob, job_id)
        if row is None:
            raise HTTPException(404, "Maintenance job not found")
        if row.status != "failed":
            raise HTTPException(409, "Only failed maintenance jobs can be retried")
        # Retry only the items that failed; finished items would repeat AI calls for nothing.
        try:
            errors = json.loads(row.errors_json or "{}")
            items = json.loads(row.items_json or "[]")
        except json.JSONDecodeError:
            errors, items = {}, []
        if isinstance(errors, dict) and errors and "job" not in errors:
            failed = [item for item in items if (f"{item.get('kind')}:{item.get('id')}" if isinstance(item, dict) else f"chunk:{item}") in errors]
            if failed:
                row.items_json = json.dumps(failed)
                row.progress_total = len(failed)
        row.status = "queued"
        row.progress_current = 0
        row.checkpoint = None
        row.errors_json = None
        return {"id": row.id, "status": row.status}


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
                .where(in_space(course_id))
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
                ConceptRelation.evidence_json.is_not(None),
            ).order_by(ConceptRelation.id)
        ).scalars().all()
        course_chunk_ids = {
            row[0]
            for row in db.execute(
                select(Chunk.id)
                .join(Document, Chunk.document_id == Document.id)
                .join(Source, Document.source_id == Source.id)
                .where(in_space(course_id), Chunk.chunk_kind == "leaf")
            ).all()
        }
        supported_relations = []
        for relation in relations:
            try:
                evidence = json.loads(relation.evidence_json or "{}")
                evidence_ids = [int(chunk_id) for chunk_id in evidence.get("chunk_ids", [])]
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            evidence_ids = [chunk_id for chunk_id in evidence_ids if chunk_id in course_chunk_ids]
            if evidence_ids:
                supported_relations.append((relation, evidence_ids))
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
                    "evidence_chunk_ids": evidence_ids,
                }
                for r, evidence_ids in supported_relations
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
        return answer(
            db, provider(), req.question, req.group_id, req.top_k, req.course_id,
            history=[turn.model_dump() for turn in req.history],
            retrieval_scope=req.retrieval_scope,
        )


@app.get("/syllabus")
def syllabus(course_id: int | None = None, regenerate: bool = False):
    """The saved study plan and its status. Never calls the AI: plans are built in the
    background when their content changes, or when `regenerate` asks for a rebuild."""
    with session_scope() as db:
        result = read_plan(db, course_id, regenerate=regenerate)
        plan = db.execute(select(StudyPlan).where(StudyPlan.scope_key == _study_scope(course_id))).scalar_one_or_none()
        result["sessions"] = session_payload(db, plan)  # how the learner's AI arranged it, or None
        return result


@app.post("/study-pack")
def study_pack(req: StudyPackRequest):
    with session_scope() as db:
        plan_scope = f"course:{req.course_id}" if req.course_id is not None else "all"
        plan = db.execute(select(StudyPlan).where(StudyPlan.scope_key == plan_scope)).scalar_one_or_none()
        if req.topic_id:
            node = db.execute(select(StudyPlanNode).where(StudyPlanNode.plan_id == plan.id, StudyPlanNode.node_id == req.topic_id)).scalar_one_or_none() if plan else None
            if node is None:
                raise HTTPException(404, "Study Plan topic not found")
            if node.node_type != "study":
                raise HTTPException(422, "Outline topics cannot be prepared as study packs; choose one of their study topics")
            req.topic = node.title
        elif plan:
            outline = db.execute(select(StudyPlanNode).where(StudyPlanNode.plan_id == plan.id, StudyPlanNode.node_type == "outline", func.lower(StudyPlanNode.title) == req.topic.strip().lower())).scalar_one_or_none()
            if outline:
                raise HTTPException(422, "Outline topics cannot be prepared as study packs; choose one of their study topics")
        scope_course = None if req.retrieval_scope == "all_sources" else req.course_id
        text_value, refs = build_study_pack_prompt(db, provider(), scope_course, req.topic, req.top_k, req.retrieval_scope)
        return {"topic": req.topic, "study_text": text_value, "sources": refs, "copy_paste_ready": True}


@app.post("/study-pack/export")
def export_study_pack(req: ExternalStudyPackRequest):
    with session_scope() as db:
        plan_scope = f"course:{req.course_id}" if req.course_id is not None else "all"
        plan = db.execute(select(StudyPlan).where(StudyPlan.scope_key == plan_scope)).scalar_one_or_none()
        if req.topic_id:
            node = db.execute(select(StudyPlanNode).where(StudyPlanNode.plan_id == plan.id, StudyPlanNode.node_id == req.topic_id)).scalar_one_or_none() if plan else None
            if node is None:
                raise HTTPException(404, "Study Plan topic not found")
            if node.node_type != "study":
                raise HTTPException(422, "Outline topics cannot be exported as study packs")
            req.topic = node.title
        elif plan:
            outline = db.execute(select(StudyPlanNode).where(StudyPlanNode.plan_id == plan.id, StudyPlanNode.node_type == "outline", func.lower(StudyPlanNode.title) == req.topic.strip().lower())).scalar_one_or_none()
            if outline:
                raise HTTPException(422, "Outline topics cannot be exported as study packs")
        text_value, refs, meta = build_external_study_pack(
            db, provider(), None if req.retrieval_scope == "all_sources" else req.course_id, req.topic, req.top_k, req.tutor, req.max_input_tokens, req.retrieval_scope,
        )
        return {"topic": req.topic, "study_text": text_value, "sources": refs, "copy_paste_ready": True, **meta}


@app.post("/plan/topics/prepare")
def prepare_plan_topic(req: PrepareTopicRequest):
    """A plan topic (or a part of one) prepared for one study session: a study unit, or its parts."""
    ai = provider()
    if hasattr(ai, "retry_max"):
        # Someone is waiting: fail fast instead of backing off. A result made without the AI is
        # marked provisional and prepared again when the topic is opened again.
        ai.retry_max = 0
    for attempt in range(2):
        try:
            with topic_lock(req.course_id, req.node_id, req.size), session_scope() as db:
                result = prepare_topic(db, ai, req.course_id, req.node_id, req.size, req.tutor, retry_provisional=req.retry)
                if result["status"] in {"unit", "thin"}:  # a study session was opened: the learner continues here next time
                    remember_position(db, scope_key(req.course_id), req.node_id, SESSION_SIZES[req.size], result["title"])
                return result
        except IntegrityError:  # another process prepared the same topic at the same moment
            if attempt:
                raise HTTPException(409, "This topic is being prepared; try again in a moment.")
        except TopicNotFound:
            raise HTTPException(404, "Study Plan topic not found")
        except OutlineTopic:
            raise HTTPException(422, "Outline topics group other topics; open one of its study topics")
        except StaleTopic:
            raise HTTPException(409, "The study plan changed since this part was made. Open its parent topic again.")


@app.get("/plan/master-prompt")
def plan_master_prompt(course_id: int | None = None, tutor: Literal["general", "chatgpt", "gemini", "claude"] = "general",
                       path_id: str | None = None):
    """The prompt that sets an external AI tutor up for the whole plan. Made from the saved plan and progress; no AI call."""
    with session_scope() as db:
        try:
            return master_prompt(db, course_id, tutor, path_id)
        except NoPlan:
            raise HTTPException(404, "There is no study plan for this space yet")


class ProfileRequest(BaseModel):
    background: str = Field(default="", max_length=2000)
    goal: str = Field(default="", max_length=1000)
    practice: Literal["questions", "coding", "both"] = "questions"


@app.get("/plan/organize-request")
def plan_organize_request(course_id: int | None = None):
    """A request the learner pastes into their own AI to arrange the plan into study sessions; no AI call here."""
    with session_scope() as db:
        try:
            return organize_request(db, course_id)
        except NoTopics:
            raise HTTPException(404, "There is no study plan for this space yet")


class SkipRequest(BaseModel):
    course_id: int | None = None
    node_id: str = Field(min_length=1, max_length=100)
    skipped: bool = True


@app.post("/plan/skip")
def skip_plan_topic(req: SkipRequest):
    """Skip a plan topic or a combined session, or bring it back. Skipped ones are never offered as next
    and are not counted as studied."""
    scope = _study_scope(req.course_id)
    with session_scope() as db:
        plan = db.execute(select(StudyPlan).where(StudyPlan.scope_key == scope)).scalar_one_or_none()
        node = db.execute(select(StudyPlanNode).where(StudyPlanNode.plan_id == plan.id, StudyPlanNode.node_id == req.node_id,
                                                      StudyPlanNode.node_type == "study")).scalar_one_or_none() if plan else None
        session = next((s for s in plan_sessions(db, plan) if s.id == req.node_id and s.together), None) if plan and node is None else None
        if node is None and session is None:
            raise HTTPException(404, "Study Plan topic not found")
        set_skipped(db, scope, req.node_id, node.title if node is not None else session.title, req.skipped)
        if not req.skipped and session is not None:  # bringing a session back brings back its topics too
            for topic in session.topics:
                set_skipped(db, scope, topic, topic, False)
        return {"skipped_ids": sorted(skipped_ids(db, scope))}


class StartAndSkipRequest(BaseModel):
    course_id: int | None = None
    reply: str = Field(min_length=1, max_length=200_000)


@app.post("/plan/start-and-skip")
def plan_start_and_skip(req: StartAndSkipRequest):
    """Apply the tutor's answer about where to start and what to skip (the JSON block the master prompt asks for)."""
    with session_scope() as db:
        try:
            return apply_start_and_skip(db, req.course_id, req.reply)
        except NoTopics:
            raise HTTPException(404, "There is no study plan for this space yet")
        except ArrangementError as exc:
            raise HTTPException(422, str(exc))


class ArrangementRequest(BaseModel):
    course_id: int | None = None
    reply: str = Field(min_length=1, max_length=200_000)


@app.post("/plan/arrangement")
def save_plan_arrangement(req: ArrangementRequest):
    """Apply the study sessions the learner's AI suggested."""
    with session_scope() as db:
        try:
            return apply_reply(db, req.course_id, req.reply)
        except NoTopics:
            raise HTTPException(404, "There is no study plan for this space yet")
        except ArrangementError as exc:
            raise HTTPException(422, str(exc))


@app.delete("/plan/arrangement")
def delete_plan_arrangement(course_id: int | None = None):
    """Back to the plan's own order: every topic a session of its own."""
    with session_scope() as db:
        return {"deleted": clear_arrangement(db, course_id)}


@app.get("/profile")
def read_profile():
    """What the learner told their tutor about themselves; added to every study unit when it is copied."""
    with session_scope() as db:
        return profile_json(get_profile(db))


@app.put("/profile")
def update_profile(req: ProfileRequest):
    with session_scope() as db:
        return profile_json(save_profile(db, req.background, req.goal, req.practice))


def _study_scope(course_id: int | None) -> str:
    return f"course:{course_id}" if course_id is not None else "all"


@app.get("/study-progress")
def study_progress(course_id: int | None = None):
    with session_scope() as db:
        scope = _study_scope(course_id)
        rows = db.execute(select(StudyProgress).where(
            StudyProgress.scope_key == scope, StudyProgress.completed.is_(True)
        )).scalars().all()
        done = {row.node_id for row in rows if row.node_id}
        topic_of = plan_topic_of(db, scope)  # part id -> its plan topic
        sizes: dict[str, dict[int, int]] = {}  # plan topic -> {session size: studied parts}
        for node_id, budget in db.execute(select(TopicDetail.node_id, TopicDetail.budget_tokens).where(
                TopicDetail.scope_key == scope, TopicDetail.node_id.in_(done))).all() if done else []:
            if node_id in topic_of:
                sizes.setdefault(topic_of[node_id], {}).setdefault(budget, 0)
                sizes[topic_of[node_id]][budget] += 1
        partly = sorted({topic_of[i] for i in done if i in topic_of} - done)
        return {"completed_topics": [row.topic_label for row in rows],
                "completed_topic_ids": [row.node_id for row in rows if row.node_id],
                # Saved before topics had ids: these can only be matched by title.
                "completed_legacy_topics": [row.topic_label for row in rows if not row.node_id],
                # Plan topics with some of their parts studied, but not all.
                "partly_done_topic_ids": partly,
                # The session size most of its studied parts were made for: reopening it there continues it.
                "partly_done_budgets": {topic: max(sizes[topic], key=lambda budget: sizes[topic][budget]) for topic in partly if topic in sizes},
                # The session opened last and not yet studied: {node_id, title, trail, budget_tokens}, or None
                # (budget_tokens 0: a start the learner chose without opening it yet).
                "position": current_position(db, scope),
                # Topics and combined sessions the learner skipped.
                "skipped_ids": sorted(skipped_ids(db, scope))}


@app.post("/study-progress")
def update_study_progress(req: StudyProgressRequest):
    scope = _study_scope(req.course_id)
    with session_scope() as db:
        plan = db.execute(select(StudyPlan).where(StudyPlan.scope_key == scope)).scalar_one_or_none()
        before = done_sessions(db, scope, plan)
        save_progress(db, scope, req.topic, req.topic_id, req.completed)
        if req.topic_id:
            roll_up(db, scope, req.topic_id, req.completed)
        sync_progress(db, scope, plan, before)  # topics studied together and their session stay in step
        return {"topic": req.topic, "completed": req.completed}


@app.get("/courses/{course_id}/stats")
def course_stats(course_id: int):
    with session_scope() as db:
        sources = db.execute(select(Source).where(in_space(course_id))).scalars().all()
        unique_youtube: dict[str, Source] = {}
        other_sources: list[Source] = []
        for source in sources:
            if source.source_type != "youtube":
                other_sources.append(source)
                continue
            key = canonical_youtube_url(source.uri or source.name)
            current = unique_youtube.get(key)
            if current is None or (source.status == "done" and current.status != "done"):
                unique_youtube[key] = source
        sources = other_sources + list(unique_youtube.values())
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
                .where(in_space(course_id))
            ).all()
        }
        relation_count = db.execute(
            select(ConceptRelation.id)
            .where(
                ConceptRelation.from_concept_id.in_(concept_ids),
                ConceptRelation.to_concept_id.in_(concept_ids),
                ConceptRelation.evidence_json.is_not(None),
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
