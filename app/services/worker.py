import json
import logging
import re
import threading
import time
from datetime import datetime, timedelta, UTC
from pathlib import Path
from sqlalchemy import func, select, update
from ..config import settings
from ..db import session_scope
from ..models import Source, Document, Chunk, Job, Group, Course, MaintenanceJob
from ..providers.factory import provider
from ..providers.base import ProviderAuthenticationError, ProviderQuotaExceededError, ProviderRateLimitError
from .ingestion import parse_file, chunk_text, list_youtube_entries, fetch_youtube_transcript, TranscriptUnavailable
from .grouping import choose_group, extract_group_concepts, merge_centroid
from .knowledge_graph import add_sequence_relations, extract_source_relations
from .agent import set_plan
from .reliability import is_retryable_error, profile_needs_refresh

NO_TEXT_MESSAGES = {
    "youtube": "None of these videos have captions LearnOS can read. Paste a transcript for each video in the Library.",
    "default": "No readable text was found. If this is a scanned PDF or image-only slides, paste the text or a transcript in the Library.",
}


def concise_space_name(value: str, limit: int = 60) -> str:
    """Turn a course title, topic or goal into a short study-space name."""
    name = re.sub(r"\s+", " ", str(value or "")).strip()
    # Playlist items are often titled "1: Introduction to ..."; the number is not part of the subject.
    name = re.sub(r"^(?:lecture|lec|part|week|module|unit|session)?\s*\d+\s*[:.)\-\u2013]\s*", "", name, flags=re.IGNORECASE)
    if len(name) <= limit:
        return name
    return name[:limit].rsplit(" ", 1)[0].rstrip(",;:-\u2013 ") or name[:limit]


class Worker(threading.Thread):
    daemon = True

    def __init__(self):
        super().__init__(name="learning-agent-worker")
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=5)

    def run(self):
        with session_scope() as db:
            stale_before = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=30)
            # Recover work interrupted by shutdown. Rate-limit retries are bounded;
            # requeue any previously scheduled retry from the older indefinite policy.
            for job in db.execute(select(Job).where(Job.status.in_(["running", "retrying"]), Job.updated_at < stale_before)).scalars().all():
                if job.status == "retrying" and self._is_rate_limit_error(job.error or ""):
                    job.retry_count = 0
                job.status = "queued"
            for job in db.execute(select(Job).where(Job.status == "retrying", Job.updated_at >= stale_before)).scalars().all():
                if self._is_rate_limit_error(job.error or ""):
                    job.retry_count = 0
                job.status = "queued"
            for job in db.execute(select(MaintenanceJob).where(MaintenanceJob.status == "running", MaintenanceJob.updated_at < stale_before)).scalars().all():
                job.status = "queued"
        while not self.stop_event.is_set():
            maintenance_id = self._claim_maintenance_job()
            if maintenance_id is not None:
                try:
                    self._process_maintenance_job(maintenance_id)
                except Exception as exc:
                    logging.exception("Maintenance job %s failed", maintenance_id)
                    with session_scope() as db:
                        maintenance = db.get(MaintenanceJob, maintenance_id)
                        if maintenance:
                            maintenance.status = "failed"
                            maintenance.errors_json = json.dumps({"job": str(exc)[:1000]}, ensure_ascii=False)
                continue
            with session_scope() as db:
                from .queue import claim_next
                job = claim_next(db)
            if not job:
                time.sleep(settings.poll_seconds)
                continue
            try:
                self.process_job(job.id)
            except Exception as exc:
                retry_count = self._record_failure(job.id, exc)
                if retry_count is None:
                    continue
                if retry_count <= settings.retry_max:
                    time.sleep(min(2 ** max(retry_count - 1, 0), 30))
                    with session_scope() as db:
                        retrying = db.get(Job, job.id)
                        if retrying and retrying.status == "retrying":
                            retrying.status = "queued"

    @staticmethod
    def _is_rate_limit_error(exc: Exception | str) -> bool:
        return isinstance(exc, ProviderRateLimitError) or "429" in str(exc).lower() or "too many requests" in str(exc).lower()

    def _claim_maintenance_job(self) -> int | None:
        with session_scope() as db:
            candidate = db.execute(select(MaintenanceJob.id).where(MaintenanceJob.status == "queued").order_by(MaintenanceJob.id).limit(1)).scalar_one_or_none()
            if candidate is None:
                return None
            now = datetime.now(UTC).replace(tzinfo=None)
            claimed = db.execute(update(MaintenanceJob).where(MaintenanceJob.id == candidate, MaintenanceJob.status == "queued").values(status="running", updated_at=now))
            if claimed.rowcount != 1:
                db.rollback()
                return None
            return candidate

    def _process_maintenance_job(self, job_id: int):
        """Resumable, single-claimer backfill for older semantic-group gaps."""
        ai = provider()
        with session_scope() as db:
            job = db.get(MaintenanceJob, job_id)
            if not job:
                return
            item_ids = json.loads(job.items_json or "[]")
            offset = job.progress_current
        for index in range(offset, len(item_ids)):
            if self.stop_event.is_set():
                with session_scope() as db:
                    job = db.get(MaintenanceJob, job_id)
                    if job:
                        job.status = "queued"
                return
            item = item_ids[index]
            if isinstance(item, dict):
                item_kind, item_id = item.get("kind"), int(item.get("id"))
            else:
                item_kind, item_id = "chunk", int(item)
            item_key = f"{item_kind}:{item_id}"
            try:
                with session_scope() as db:
                    if item_kind == "chunk":
                        chunk = db.get(Chunk, item_id)
                        if chunk and (chunk.group_id is None or chunk.classification_status != "done") and chunk.embedding_status == "done" and chunk.embedding_json:
                            doc = db.get(Document, chunk.document_id)
                            source = db.get(Source, doc.source_id) if doc else None
                            if source and source.status == "done":
                                embedding = json.loads(chunk.embedding_json)
                                group, confidence = choose_group(db, ai, chunk, embedding, self._source_hint(source), settings.group_similarity_threshold)
                                chunk.group_id = group.id
                                chunk.group_confidence = confidence
                                chunk.classification_status = "done"
                                members = db.execute(select(Chunk.id).where(Chunk.group_id == group.id, Chunk.chunk_kind == "leaf", Chunk.embedding_status == "done")).all()
                                group.centroid_json = json.dumps(merge_centroid(json.loads(group.centroid_json) if group.centroid_json else None, embedding, len(members)))
                    elif item_kind == "source":
                        source = db.get(Source, item_id)
                        if source and source.status == "done":
                            try:
                                profile = json.loads(source.profile_json or "{}")
                            except json.JSONDecodeError:
                                profile = {}
                            if profile_needs_refresh(profile):
                                self._ensure_source_profile(db, source, ai)
                            if source.course_id is None and (source.assignment_mode == "auto" or item.get("auto_assign_unassigned")):
                                self._auto_group_source(db, source, ai, force=bool(item.get("auto_assign_unassigned")))
                    job = db.get(MaintenanceJob, job_id)
                    job.progress_current = index + 1
                    job.checkpoint = item_key
            except Exception as exc:
                logging.exception("Maintenance job %s failed for %s", job_id, item_key)
                with session_scope() as db:
                    job = db.get(MaintenanceJob, job_id)
                    errors = json.loads(job.errors_json or "{}")
                    errors[item_key] = str(exc)[:1000]
                    job.errors_json = json.dumps(errors, ensure_ascii=False)
                    job.progress_current = index + 1
                    job.checkpoint = item_key
            if not self.stop_event.is_set():
                continue
        with session_scope() as db:
            job = db.get(MaintenanceJob, job_id)
            if job:
                job.status = "failed" if job.errors_json and json.loads(job.errors_json) else "done"

    def _auto_group_source(self, db, source: Source, ai, *, force: bool = False):
        """Assign an unfiled source to a matching learning space or create one, recording why."""
        if (source.assignment_mode != "auto" and not force) or source.course_id is not None:
            return
        try:
            profile = json.loads(source.profile_json or "{}")
        except json.JSONDecodeError:
            profile = {}
        courses = db.execute(select(Course).order_by(Course.name)).scalars().all()
        course_id = None
        new_name = ""
        reason = ""
        confidence = 0.0
        if courses:
            schema = {
                "type": "object",
                "properties": {
                    "course_id": {"type": "integer"},
                    "new_course_name": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["course_id", "new_course_name", "confidence", "reason"],
            }
            prompt = f"""File this learning source into an existing learning space only when its title, description, domain, and outline fit the source's learning goal and outcomes. Otherwise propose a concise new space.
Use the structured source profile and indexed content as evidence. A shared broad domain alone is not a fit. Be conservative; confidence must be at least 0.65 to choose an existing space.
Source profile: {json.dumps(profile, ensure_ascii=False)}
Available spaces: {json.dumps([{"id": course.id, "name": course.name, "description": course.description, "domain": course.domain, "outline": course.outline} for course in courses], ensure_ascii=False)}
Return an available course_id only when the match is strong and confidence >= 0.65. Otherwise return course_id 0 and a short new_course_name (2-6 words, a subject name, not a sentence) based on the source's specific learning objective.
Give a one-sentence reason a learner can read."""
            try:
                result = ai.generate_json(prompt, task="classification", schema=schema)
                course_id = int(result.get("course_id") or 0) or None
                confidence = float(result.get("confidence") or 0.0)
                new_name = concise_space_name(str(result.get("new_course_name") or ""))
                reason = str(result.get("reason") or "").strip()[:400]
                if confidence < 0.65 or course_id not in {course.id for course in courses}:
                    course_id = None
            except Exception as exc:
                if is_retryable_error(exc):
                    raise
                logging.exception("AI source-to-course classification failed; using profile fallback")
        if course_id is not None:
            matched = next(course for course in courses if course.id == course_id)
            explanation = f'Filed into "{matched.name}" automatically ({confidence:.0%} match).'
        else:
            suggested_groups = profile.get("suggested_group_names") or []
            suggested_topics = profile.get("topics") or []
            if not isinstance(suggested_groups, list):
                suggested_groups = [suggested_groups]
            if not isinstance(suggested_topics, list):
                suggested_topics = [suggested_topics]
            # Subject names first; the learning goal is a sentence and only a last resort.
            suggestions = [value for value in [profile.get("course_title"), *suggested_groups, *suggested_topics, profile.get("domain"), profile.get("primary_learning_goal")] if value]
            if not new_name and suggestions:
                new_name = concise_space_name(str(suggestions[0]))
            if not new_name:
                source.assignment_status = "unassigned"
                source.assignment_reason = "Not enough source metadata to infer a learning space. Choose one manually in the library."
                return
            normalized = new_name.casefold()
            target = next((course for course in courses if course.name.casefold() == normalized), None)
            if target is None:
                target = Course(name=new_name, description=str(profile.get("description") or "")[:2000], domain=str(profile.get("domain") or "")[:255] or None)
                db.add(target)
                db.flush()
                explanation = f'Created a new space "{new_name}" because no existing space matched this source.'
            else:
                explanation = f'Filed into "{target.name}", the space with the same subject name.'
            course_id = target.id
        source.course_id = course_id
        if force:
            source.assignment_mode = "auto"
        source.assignment_status = "assigned"
        source.assignment_reason = f"{explanation} {reason}".strip()[:1000]

    def _record_failure(self, job_id: int, exc: Exception) -> int | None:
        with session_scope() as db:
            job = db.get(Job, job_id)
            if not job:
                return None
            job.retry_count += 1
            job.error = str(exc)
            job.error_message = self._user_error(job, exc)
            src = db.get(Source, job.source_id)
            if isinstance(exc, ProviderAuthenticationError):
                job.status = "failed"
                if src:
                    src.status = "failed"
                    src.error = str(exc)
                return None
            if isinstance(exc, ProviderQuotaExceededError):
                job.status = "failed"
                if src:
                    src.status = "failed"
                    src.error = str(exc)
                return None
            if self._is_rate_limit_error(exc):
                if job.retry_count <= settings.retry_max:
                    job.status = "retrying"
                    if src:
                        src.status = "queued"
                else:
                    job.status = "failed"
                    if src:
                        src.status = "failed"
                        src.error = str(exc)
            elif job.retry_count <= settings.retry_max:
                job.status = "retrying"
                if src:
                    src.status = "queued"
            else:
                job.status = "failed"
                if src:
                    src.status = "failed"
                    src.error = str(exc)
            return job.retry_count

    @staticmethod
    def _user_error(job: Job, exc: Exception) -> str:
        if isinstance(exc, ProviderAuthenticationError):
            return "AI provider authentication failed. Check GEMINI_API_KEY in .env, or clear it to use local processing."
        if isinstance(exc, ProviderQuotaExceededError):
            return f"Gemini reports that the daily quota for {job.stage.replace('_', ' ')} is exhausted or unavailable. Automatic retries stopped. Daily quotas reset at midnight Pacific; if the limit is unavailable or zero, check the Gemini project/model quota. Choose Resume after recovery, or switch provider/model."
        if isinstance(exc, ProviderRateLimitError) or "429" in str(exc):
            return f"Gemini temporary rate limit during {job.stage.replace('_', ' ')}. LearnOS will retry automatically ({job.retry_count}/{settings.retry_max}); if it persists, processing stops and Resume remains available."
        if job.stage == "transcript":
            return "Could not retrieve captions from YouTube. This is usually temporary; LearnOS retries automatically, and you can choose Resume later."
        if job.stage == "structure" and "playlist" in str(exc).lower():
            return "Could not read this playlist. Check that it is public and try again."
        if job.stage == "structure":
            return "Could not prepare this YouTube source. Check the URL and video or playlist availability."
        return f"Processing failed during {job.stage.replace('_', ' ')}. Try again, or check the source and service availability."

    def _source_hint(self, src: Source) -> str | None:
        pieces = [src.topic_hint, src.domain_hint, src.description]
        if src.metadata_json:
            try:
                meta = json.loads(src.metadata_json)
                for key in ("topics", "topic", "keywords", "subject"):
                    value = meta.get(key)
                    if isinstance(value, list):
                        pieces.append(", ".join(map(str, value)))
                    elif value:
                        pieces.append(str(value))
            except json.JSONDecodeError:
                pass
        return " | ".join(dict.fromkeys(p.strip() for p in pieces if p and p.strip()))[:2500] or None

    def _ensure_chunk_hierarchy(self, db, doc: Document):
        """Create bounded parent windows and deterministic next-links around leaf chunks."""
        leaves = db.execute(
            select(Chunk)
            .where(Chunk.document_id == doc.id, Chunk.chunk_kind == "leaf")
            .order_by(Chunk.chunk_index)
        ).scalars().all()
        if not leaves:
            return

        window = max(2, settings.parent_chunk_window)
        for start in range(0, len(leaves), window):
            members = leaves[start:start + window]
            parent_index = -(start // window + 1)
            parent = db.execute(
                select(Chunk).where(
                    Chunk.document_id == doc.id,
                    Chunk.chunk_kind == "parent",
                    Chunk.chunk_index == parent_index,
                )
            ).scalar_one_or_none()
            parent_text = "\n\n".join(c.text for c in members)
            parent_context = " | ".join(
                value for value in [doc.title or "", doc.section or ""] if value
            )
            if not parent:
                parent = Chunk(
                    document_id=doc.id,
                    chunk_index=parent_index,
                    chunk_kind="parent",
                    text=parent_text,
                    contextual_text=(parent_context + "\n" + parent_text).strip() if parent_context else parent_text,
                    summary=re.sub(r"\s+", " ", parent_text).strip()[:600],
                    token_estimate=max(1, len(parent_text) // 4),
                    embedding_status="skipped",
                    classification_status="skipped",
                    cluster_status="skipped",
                )
                db.add(parent)
                db.flush()
            else:
                parent.text = parent_text
                parent.contextual_text = (parent_context + "\n" + parent_text).strip() if parent_context else parent_text
                parent.summary = re.sub(r"\s+", " ", parent_text).strip()[:600]
                parent.token_estimate = max(1, len(parent_text) // 4)
            for leaf in members:
                leaf.parent_chunk_id = parent.id

        add_sequence_relations(db, doc.id)

    def _ensure_file_documents(self, db, src: Source):
        raw_docs = parse_file(Path(src.local_path))
        for raw in raw_docs:
            doc = db.execute(
                select(Document).where(
                    Document.source_id == src.id,
                    Document.external_id == raw.external_id,
                )
            ).scalar_one_or_none()
            if not doc:
                doc = Document(
                    source_id=src.id,
                    external_id=raw.external_id,
                    title=raw.title,
                    section=raw.section,
                    status="processing",
                )
                db.add(doc)
                db.flush()
            existing_chunks = db.execute(
                select(Chunk).where(Chunk.document_id == doc.id, Chunk.chunk_kind == "leaf")
            ).scalars().all()
            existing_indexes = {c.chunk_index for c in existing_chunks}
            pieces = chunk_text(raw.text, settings.chunk_size, settings.chunk_overlap)
            for ci, text_piece in enumerate(pieces):
                if ci in existing_indexes:
                    continue
                db.add(
                    Chunk(
                        document_id=doc.id,
                        chunk_index=ci,
                        chunk_kind="leaf",
                        text=text_piece,
                        token_estimate=max(1, len(text_piece) // 4),
                    )
                )
            db.flush()
            self._ensure_chunk_hierarchy(db, doc)

    def _ensure_youtube_documents(self, db, src: Source):
        info, videos = list_youtube_entries(src.uri)
        if info.get("title") and (not src.name or src.name.startswith(("http://", "https://"))):
            src.name = info["title"][:500]
        try:
            metadata = json.loads(src.metadata_json or "{}")
        except json.JSONDecodeError:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["platform"] = {"site": "youtube", **{key: value for key, value in info.items() if value}, "video_count": len(videos)}
        src.metadata_json = json.dumps(metadata, ensure_ascii=False)
        for video in videos:
            video_metadata = json.dumps({key: value for key, value in video.items() if key != "id" and value}, ensure_ascii=False)
            doc = db.execute(
                select(Document).where(
                    Document.source_id == src.id,
                    Document.external_id == video["id"],
                )
            ).scalar_one_or_none()
            if not doc:
                doc = Document(
                    source_id=src.id,
                    external_id=video["id"],
                    title=video["title"],
                    section="youtube transcript",
                    status="pending",
                    metadata_json=video_metadata,
                )
                db.add(doc)
            else:
                # Older versions titled single videos by their ID.
                if doc.title == doc.external_id and video["title"] != video["id"]:
                    doc.title = video["title"]
                if not doc.metadata_json:
                    doc.metadata_json = video_metadata
        return len(videos)

    def _ensure_source_profile(self, db, src: Source, ai):
        try:
            current_profile = json.loads(src.profile_json or "{}")
        except json.JSONDecodeError:
            current_profile = {}
        if not profile_needs_refresh(current_profile):
            return
        chunks = db.execute(
            select(Chunk)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.source_id == src.id, Chunk.chunk_kind == "leaf")
            .order_by(Chunk.chunk_index)
            .limit(3)
        ).scalars().all()
        if not chunks:
            return
        try:
            provided = json.loads(src.metadata_json or "{}")
        except json.JSONDecodeError:
            provided = {}
        first_doc = db.execute(select(Document).where(Document.source_id == src.id).order_by(Document.id).limit(1)).scalar_one_or_none()
        document_count = db.scalar(select(func.count()).select_from(Document).where(Document.source_id == src.id)) or 0
        # A playlist's own title describes the course; its first video's title does not.
        name_is_url = (src.name or "").startswith(("http://", "https://"))
        if first_doc and first_doc.title and (document_count == 1 or name_is_url):
            explicit_title = first_doc.title
        else:
            explicit_title = src.name
        profile_prompt = f"""Profile the educational purpose and identity of this single source. Do not merge it with other sources.
Do not fill fields from general knowledge. Use an empty string or empty list when the source does not provide enough evidence. Mark fields inferred from content in inferred_fields and assign conservative confidence.
Source ID: {src.id}
Title: {explicit_title}
Source URI: {src.uri or 'unknown'}
Source type: {src.source_type}
User/source metadata (may be incomplete): {json.dumps(provided, ensure_ascii=False)}
User hints (weak evidence): {self._source_hint(src) or 'none'}
Transcript or document excerpts:
{chr(10).join(c.text[:2200] for c in chunks)}
Return JSON containing author, provider/institution, domain, subdomain, estimated_level, course_title, primary_learning_goal, learning_objectives, expected_prerequisites, learner_capabilities, downstream_applications, topics, source_role (syllabus, lecture material, textbook, chapter, tutorial, documentation, project, supplementary material, reference material, or unknown), course_identity_confidence, learning_intent_confidence, inferred_fields, unknown_fields, description, and suggested_group_names. Do not equate topic overlap with course identity.
"""
        schema = {
            "type": "object",
            "properties": {
                "author": {"type": "string"}, "institution": {"type": "string"}, "provider": {"type": "string"},
                "domain": {"type": "string"},
                "subdomain": {"type": "string"},
                "topics": {"type": "array", "items": {"type": "string"}},
                "estimated_level": {"type": "string"},
                "course_title": {"type": "string"},
                "primary_learning_goal": {"type": "string"},
                "learning_objectives": {"type": "array", "items": {"type": "string"}},
                "expected_prerequisites": {"type": "array", "items": {"type": "string"}},
                "learner_capabilities": {"type": "array", "items": {"type": "string"}},
                "downstream_applications": {"type": "array", "items": {"type": "string"}},
                "source_role": {"type": "string"},
                "course_identity_confidence": {"type": "number"},
                "learning_intent_confidence": {"type": "number"},
                "inferred_fields": {"type": "array", "items": {"type": "string"}},
                "unknown_fields": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "string"},
                "suggested_group_names": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["author", "institution", "provider", "domain", "subdomain", "estimated_level", "course_title", "primary_learning_goal", "learning_objectives", "expected_prerequisites", "learner_capabilities", "downstream_applications", "source_role", "course_identity_confidence", "learning_intent_confidence", "inferred_fields", "unknown_fields", "topics", "description", "suggested_group_names"],
        }
        try:
            profile = ai.generate_json(profile_prompt, task="classification", schema=schema)
            confidence = float(profile.get("course_identity_confidence", 0.0) or 0.0)
            web_evidence = []
            if confidence < 0.55 and (src.uri or len(explicit_title.strip()) > 8) and callable(getattr(ai, "search_grounded", None)):
                query = " ".join(part for part in (explicit_title, profile.get("author"), profile.get("institution"), src.uri) if part)
                try:
                    grounded = ai.search_grounded(f"Identify this educational source from authoritative primary pages only. Query: {query}. Return concise course title, institution, description, stated learning outcomes, target level, and stated prerequisites. Prefer official institution/provider/instructor pages. Do not infer prerequisites from general topic relationships.")
                except Exception:
                    grounded = {}
                web_evidence = grounded.get("sources", [])
                grounded_text = grounded.get("text", "")
                if grounded_text:
                    web_schema = {"type": "object", "properties": {
                        "official_course_title": {"type": "string"}, "official_provider": {"type": "string"},
                        "official_description": {"type": "string"}, "stated_outcomes": {"type": "array", "items": {"type": "string"}},
                        "target_level": {"type": "string"}, "stated_prerequisites": {"type": "array", "items": {"type": "string"}},
                    }, "required": ["official_course_title", "official_provider", "official_description", "stated_outcomes", "target_level", "stated_prerequisites"]}
                    try:
                        grounded_facts = ai.generate_json(f"Extract only facts stated by authoritative official course/provider/instructor pages from this search-grounded text. Do not use generic articles and do not infer prerequisites. Return empty fields when unsupported.\nSearch-grounded text: {grounded_text[:5000]}\nGrounded URLs: {json.dumps(web_evidence, ensure_ascii=False)}", task="classification", schema=web_schema)
                    except Exception:
                        grounded_facts = {}
                    field_map = {"course_title": "official_course_title", "institution": "official_provider", "provider": "official_provider", "description": "official_description", "learning_objectives": "stated_outcomes", "estimated_level": "target_level", "expected_prerequisites": "stated_prerequisites"}
                    conflicts = []
                    web_provenance = {}
                    for local_field, grounded_field in field_map.items():
                        candidate = grounded_facts.get(grounded_field)
                        current = profile.get(local_field)
                        has_candidate = bool(candidate) and (any(str(x).strip() for x in candidate) if isinstance(candidate, list) else bool(str(candidate).strip()))
                        if not has_candidate:
                            continue
                        if current and str(current).strip().casefold() != str(candidate).strip().casefold():
                            conflicts.append({"field": local_field, "source_interpretation": current, "web_interpretation": candidate})
                        elif not current:
                            profile[local_field] = candidate
                            web_provenance[local_field] = "official_web"
                    if conflicts:
                        profile["profile_conflicts"] = conflicts
                        profile["course_identity_confidence"] = min(confidence, float(profile.get("course_identity_confidence", 0.0) or 0.0)) * 0.65
                        profile["learning_intent_confidence"] = min(float(profile.get("learning_intent_confidence", 0.0) or 0.0), 0.6)
                    elif web_provenance:
                        profile["course_identity_confidence"] = max(float(profile.get("course_identity_confidence", 0.0) or 0.0), 0.65)
                    profile["web_field_provenance"] = web_provenance
                    profile["web_search_queries"] = grounded.get("queries", [])
            profile.update({"schema_version": 2, "source_id": src.id, "title": explicit_title, "source_type": src.source_type,
                            "course_identity_confidence": max(0.0, min(1.0, float(profile.get("course_identity_confidence", 0.0) or 0.0))),
                            "learning_intent_confidence": max(0.0, min(1.0, float(profile.get("learning_intent_confidence", 0.0) or 0.0))),
                            "field_provenance": {"title": "source_metadata", "source_type": "source_metadata", "author": "inferred" if "author" in profile.get("inferred_fields", []) else ("unknown" if not profile.get("author") else "source_content"), "institution": "inferred" if "institution" in profile.get("inferred_fields", []) else ("unknown" if not profile.get("institution") else "source_content"), "course_title": "inferred" if "course_title" in profile.get("inferred_fields", []) else ("unknown" if not profile.get("course_title") else "source_content")},
                            "web_evidence": web_evidence})
            src.profile_json = json.dumps(profile, ensure_ascii=False)
        except Exception as exc:
            # A temporary outage must fail the job so it is retried. Saving the empty
            # fallback as a finished profile would lock the source out of organisation.
            if is_retryable_error(exc):
                raise
            logging.exception("Source profiling failed for source %s; saved a degraded profile that will be retried", src.id)
            src.profile_json = json.dumps(
                {"schema_version": 2, "degraded": True, "source_id": src.id, "title": explicit_title, "source_type": src.source_type,
                 "domain": src.domain_hint or "", "subdomain": "", "topics": [src.topic_hint] if src.topic_hint else [],
                 "estimated_level": "", "course_title": "", "primary_learning_goal": "", "learning_objectives": [],
                 "expected_prerequisites": [], "learner_capabilities": [], "downstream_applications": [], "source_role": "unknown",
                 "course_identity_confidence": 0.0, "learning_intent_confidence": 0.0, "inferred_fields": [],
                 "unknown_fields": ["author", "institution", "course_title", "primary_learning_goal"],
                 "field_provenance": {"author": "unknown", "institution": "unknown", "course_title": "unknown"}, "web_evidence": [],
                 "description": "", "suggested_group_names": []},
                ensure_ascii=False,
            )

    def _prepare_contextual_text(self, db, src: Source, chunk: Chunk, ai=None, enable_llm=False):
        if chunk.contextual_text:
            return
        doc = db.get(Document, chunk.document_id)
        prefix = [src.name]
        if doc and doc.title:
            prefix.append(doc.title)
        if doc and doc.section:
            prefix.append(doc.section)
        if src.domain_hint:
            prefix.append(src.domain_hint)
        if src.topic_hint:
            prefix.append(src.topic_hint)
        if src.description:
            prefix.append(src.description[:500])
        static_context = " | ".join(prefix)
        if enable_llm and ai is not None:
            prompt = f"""Give a very short context sentence that situates the following chunk within the learning source.
Do not add new facts. Return only the context sentence.
Source: {static_context}
Chunk:
{chunk.text[:4500]}
"""
            try:
                llm_context = ai.generate_text(prompt, task="contextualization").strip()
                if llm_context:
                    static_context = f"{static_context} | {llm_context[:800]}"
            except Exception:
                pass
        chunk.contextual_text = static_context + "\n" + chunk.text

    def process_job(self, job_id: int):
        ai = provider()
        with session_scope() as db:
            job = db.get(Job, job_id)
            src = db.get(Source, job.source_id) if job else None
            if not job or not src:
                raise ValueError("Job/source not found")
            set_plan(job, src)
            src.status = "processing"
            job.error = None
            job.error_message = None
            job.stage = "structure"
            if src.source_type in {"file", "transcript"}:
                self._ensure_file_documents(db, src)
            elif src.source_type == "youtube":
                total_docs = self._ensure_youtube_documents(db, src)
                job.progress_total = total_docs
            else:
                raise ValueError(f"unsupported source type {src.source_type}")
            self._ensure_source_profile(db, src, ai)
            db.commit()

        if src.source_type == "youtube":
            with session_scope() as db:
                job = db.get(Job, job_id)
                src = db.get(Source, job.source_id)
                docs = db.execute(
                    select(Document).where(Document.source_id == src.id).order_by(Document.id)
                ).scalars().all()
                job.stage = "transcript"
                job.progress_total = len(docs)
                doc_ids = [d.id for d in docs]
                db.commit()
            # One video without captions must not block the rest of a playlist. Videos with
            # no usable captions wait for a pasted transcript; a temporary YouTube block stops
            # fetching for this run and leaves the remaining videos for a retry.
            blocked: Exception | None = None
            for i, doc_id in enumerate(doc_ids):
                with session_scope() as db:
                    job = db.get(Job, job_id)
                    doc = db.get(Document, doc_id)
                    existing = db.execute(
                        select(Chunk.id).where(
                            Chunk.document_id == doc.id,
                            Chunk.chunk_kind == "leaf",
                        )
                    ).first()
                    if not existing and doc.status != "needs_input":
                        text_value = ""
                        if blocked is not None:
                            doc.status = "failed"
                            doc.error = "Not fetched because YouTube was refusing requests. Retry later."
                        else:
                            try:
                                text_value = fetch_youtube_transcript(doc.external_id)
                            except TranscriptUnavailable as exc:
                                doc.status = "needs_input"
                                doc.error = exc.user_message
                            except Exception as exc:
                                logging.warning("Transcript fetch failed for %s: %s", doc.external_id, exc)
                                blocked = exc
                                doc.status = "failed"
                                doc.error = "YouTube did not return the transcript (temporary). Retry later."
                        pieces = chunk_text(text_value, settings.chunk_size, settings.chunk_overlap) if text_value else []
                        if text_value and not pieces:
                            doc.status = "needs_input"
                            doc.error = "The captions for this video are empty."
                        for ci, piece in enumerate(pieces):
                            db.add(
                                Chunk(
                                    document_id=doc.id,
                                    chunk_index=ci,
                                    chunk_kind="leaf",
                                    text=piece,
                                    token_estimate=max(1, len(piece) // 4),
                                )
                            )
                        if pieces:
                            doc.status = "processing"
                            doc.error = None
                    elif existing:
                        doc.status = "processing"
                    db.flush()
                    self._ensure_chunk_hierarchy(db, doc)
                    job.progress_current = i + 1
                    job.checkpoint = f"video:{i + 1}/{len(doc_ids)}"
                    db.commit()
            if blocked is not None:
                with session_scope() as db:
                    has_text = db.execute(
                        select(Chunk.id).join(Document, Chunk.document_id == Document.id)
                        .where(Document.source_id == src.id, Chunk.chunk_kind == "leaf").limit(1)
                    ).first()
                if not has_text:
                    # Nothing usable yet and the cause is temporary: let the job retry.
                    raise RuntimeError(f"Transcript fetching was refused by YouTube: {blocked}")

        with session_scope() as db:
            job = db.get(Job, job_id)
            src = db.get(Source, job.source_id)
            self._ensure_source_profile(db, src, ai)
            db.commit()

        with session_scope() as db:
            job = db.get(Job, job_id)
            src = db.get(Source, job.source_id)
            chunk_rows = db.execute(
                select(Chunk)
                .join(Document, Chunk.document_id == Document.id)
                .where(
                    Document.source_id == src.id,
                    Chunk.chunk_kind == "leaf",
                )
                .order_by(Chunk.id)
            ).scalars().all()
            if not chunk_rows:
                # Finishing as "done" would list an empty source as ready to study.
                message = NO_TEXT_MESSAGES.get(src.source_type, NO_TEXT_MESSAGES["default"])
                src.status = "needs_input"
                src.error = message
                job.status = "done"
                job.stage = "done"
                job.checkpoint = "needs_input"
                job.error_message = message[:500]
                db.commit()
                return
            job.stage = "index"
            job.progress_current = 0
            job.progress_total = len(chunk_rows)
            db.commit()
            chunk_ids = [c.id for c in chunk_rows]

        contextualize_enabled = (
            settings.contextual_indexing
            and len(chunk_ids) <= settings.contextual_indexing_max_chunks
        )

        for batch_start in range(0, len(chunk_ids), settings.embedding_batch_size):
            batch_ids = chunk_ids[batch_start:batch_start + settings.embedding_batch_size]
            with session_scope() as db:
                job = db.get(Job, job_id)
                chunks = [db.get(Chunk, cid) for cid in batch_ids]
                chunks = [c for c in chunks if c is not None]
                src = db.get(Source, job.source_id)
                pending = [c for c in chunks if c.embedding_status != "done"]
                for chunk in chunks:
                    self._prepare_contextual_text(db, src, chunk, ai, contextualize_enabled)
                    if chunk.summary is None:
                        chunk.summary = re.sub(r"\s+", " ", chunk.text).strip()[:400]
                if pending:
                    vectors = ai.embed([c.contextual_text or c.text for c in pending])
                    if len(vectors) != len(pending):
                        raise RuntimeError("Embedding provider returned a mismatched batch size")
                    for chunk, emb in zip(pending, vectors):
                        chunk.embedding_json = json.dumps(emb)
                        chunk.embedding_status = "done"
                # Persist embeddings before semantic organization. On resume,
                # completed embedding batches are skipped if grouping fails.
                job.stage = "organize"
                db.commit()

            with session_scope() as db:
                job = db.get(Job, job_id)
                chunks = [db.get(Chunk, cid) for cid in batch_ids]
                chunks = [c for c in chunks if c is not None]
                src = db.get(Source, job.source_id)
                for chunk in chunks:
                    if chunk.classification_status != "done":
                        emb = json.loads(chunk.embedding_json)
                        group, confidence = choose_group(
                            db, ai, chunk, emb, self._source_hint(src), settings.group_similarity_threshold
                        )
                        chunk.group_id = group.id
                        chunk.group_confidence = confidence
                        chunk.classification_status = "done"
                        members = db.execute(
                            select(Chunk.id).where(
                                Chunk.group_id == group.id,
                                Chunk.chunk_kind == "leaf",
                                Chunk.embedding_status == "done",
                            )
                        ).all()
                        group.centroid_json = json.dumps(
                            merge_centroid(
                                json.loads(group.centroid_json) if group.centroid_json else None,
                                emb,
                                len(members),
                            )
                        )
                job.progress_current = min(batch_start + len(batch_ids), len(chunk_ids))
                job.checkpoint = f"chunk:{job.progress_current}/{len(chunk_ids)}"
                job.stage = "index"
                db.commit()

        with session_scope() as db:
            job = db.get(Job, job_id)
            src = db.get(Source, job.source_id)
            group_ids = [
                row[0]
                for row in db.execute(
                    select(Chunk.group_id)
                    .join(Document, Chunk.document_id == Document.id)
                    .where(
                        Document.source_id == src.id,
                        Chunk.chunk_kind == "leaf",
                        Chunk.group_id.is_not(None),
                    )
                    .distinct()
                ).all()
            ]
            job.stage = "concepts"
            job.progress_current = 0
            job.progress_total = len(group_ids)
            db.commit()

        for i, group_id in enumerate(group_ids, 1):
            with session_scope() as db:
                graph_job = db.get(Job, job_id)
                if graph_job:
                    graph_job.stage = "graph"
                job = db.get(Job, job_id)
                group = db.get(Group, group_id)
                if group:
                    extract_group_concepts(db, ai, group)
                    db.query(Chunk).filter(
                        Chunk.group_id == group.id,
                        Chunk.chunk_kind == "leaf",
                    ).update({"cluster_status": "done"}, synchronize_session=False)
                job.progress_current = i
                job.checkpoint = f"group:{i}/{len(group_ids)}"
                db.commit()

        with session_scope() as db:
            graph_job = db.get(Job, job_id)
            if graph_job:
                graph_job.stage = "graph"
                graph_job.checkpoint = "graph:source-relations"
                extract_source_relations(db, ai, graph_job.source_id)

        with session_scope() as db:
            job = db.get(Job, job_id)
            src = db.get(Source, job.source_id)
            if src.assignment_mode == "auto":
                self._auto_group_source(db, src, ai)
            elif src.course_id is not None:
                src.assignment_status = "assigned"
            else:
                src.assignment_status = "unassigned"
            indexed_doc_ids = set(db.execute(
                select(Chunk.document_id).join(Document, Chunk.document_id == Document.id)
                .where(Document.source_id == src.id, Chunk.chunk_kind == "leaf").distinct()
            ).scalars().all())
            for doc in db.execute(select(Document).where(Document.source_id == src.id)).scalars().all():
                if doc.id in indexed_doc_ids:
                    doc.status = "done"
            src.status = "done"
            src.error = None
            job.status = "done"
            job.stage = "done"
            job.progress_current = job.progress_total
            job.checkpoint = "done"
            db.commit()
