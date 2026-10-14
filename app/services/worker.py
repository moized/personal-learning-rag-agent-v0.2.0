import json
import math
import re
import threading
import time
from pathlib import Path
from sqlalchemy import select, func
from ..config import settings
from ..db import session_scope
from ..models import Source, Document, Chunk, Job
from ..providers.base import DecisionPending, is_automatic
from ..providers.factory import provider
from .ingestion import parse_file, chunk_text, list_youtube_videos, fetch_youtube_transcript
from .grouping import organize_batch
from .dedupe import mark_duplicates
from .decisions import pause_job, requeue_waiting_jobs
from .agent import set_plan


class Worker(threading.Thread):
    daemon = True

    def __init__(self):
        super().__init__(name="learning-agent-worker")
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        with session_scope() as db:
            for job in db.execute(select(Job).where(Job.status == "running")).scalars().all():
                job.status = "queued"
            # Answers pasted while the worker was down still resume their jobs.
            requeue_waiting_jobs(db)
        while not self.stop_event.is_set():
            with session_scope() as db:
                from .queue import claim_next
                job = claim_next(db)
            if not job:
                time.sleep(settings.poll_seconds)
                continue
            try:
                self.process_job(job.id)
            except DecisionPending as pending:
                # Not a failure: the job sleeps until the user pastes the external AI's answer.
                with session_scope() as db:
                    pause_job(db, job.id, pending)
            except Exception as exc:
                with session_scope() as db:
                    job = db.get(Job, job.id)
                    if not job:
                        continue
                    job.retry_count += 1
                    job.error = str(exc)
                    if job.retry_count <= settings.retry_max:
                        job.status = "queued"
                        src = db.get(Source, job.source_id)
                        if src:
                            src.status = "queued"
                    else:
                        job.status = "failed"
                        src = db.get(Source, job.source_id)
                        if src:
                            src.status = "failed"
                            src.error = str(exc)
                time.sleep(min(2 ** max(job.retry_count - 1, 0), 30))

    @staticmethod
    def _organize_counts(db, own) -> tuple[int, int]:
        return tuple(db.execute(
            select(func.count(Chunk.id).filter(Chunk.classification_status == "done"), func.count(Chunk.id))
            .join(Document, Chunk.document_id == Document.id).where(*own)
        ).one())

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

    def _ensure_file_documents(self, db, src: Source):
        raw_docs = parse_file(Path(src.local_path))
        for raw in raw_docs:
            doc = db.execute(select(Document).where(Document.source_id == src.id, Document.external_id == raw.external_id)).scalar_one_or_none()
            if not doc:
                doc = Document(source_id=src.id, external_id=raw.external_id, title=raw.title, section=raw.section, status="processing")
                db.add(doc)
                db.flush()
            existing_chunks = db.execute(select(Chunk).where(Chunk.document_id == doc.id)).scalars().all()
            existing_indexes = {c.chunk_index for c in existing_chunks}
            pieces = chunk_text(raw.text, settings.chunk_size, settings.chunk_overlap)
            for ci, text_piece in enumerate(pieces):
                if ci in existing_indexes:
                    continue
                db.add(Chunk(document_id=doc.id, chunk_index=ci, text=text_piece, token_estimate=max(1, len(text_piece) // 4)))

    def _ensure_youtube_documents(self, db, src: Source):
        videos = list_youtube_videos(src.uri)
        for video_id, title in videos:
            doc = db.execute(select(Document).where(Document.source_id == src.id, Document.external_id == video_id)).scalar_one_or_none()
            if not doc:
                doc = Document(source_id=src.id, external_id=video_id, title=title, section="youtube transcript", status="pending")
                db.add(doc)
        return len(videos)

    def _ensure_source_profile(self, db, src: Source, ai):
        if src.profile_json or not is_automatic(ai):
            return
        chunks = db.execute(
            select(Chunk).join(Document, Chunk.document_id == Document.id)
            .where(Document.source_id == src.id).order_by(Chunk.chunk_index).limit(3)
        ).scalars().all()
        if not chunks:
            return
        profile_prompt = f"""Create a compact learning-source profile from this material.
Hints are weak signals. Infer the actual subject from content.
Name: {src.name}
Hints: {self._source_hint(src) or 'none'}
Preview:
{chr(10).join(c.text[:1800] for c in chunks)}
Return JSON with domain, topics (array), level, description, and suggested_group_names (array).
"""
        schema = {
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
                "topics": {"type": "array", "items": {"type": "string"}},
                "level": {"type": "string"},
                "description": {"type": "string"},
                "suggested_group_names": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["domain", "topics", "level", "description", "suggested_group_names"],
        }
        try:
            src.profile_json = json.dumps(ai.generate_json(profile_prompt, task="classification", schema=schema), ensure_ascii=False)
        except Exception:
            src.profile_json = json.dumps({"domain": src.domain_hint, "topics": [src.topic_hint] if src.topic_hint else []}, ensure_ascii=False)

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
                job = db.get(Job, job_id); src = db.get(Source, job.source_id)
                docs = db.execute(select(Document).where(Document.source_id == src.id).order_by(Document.id)).scalars().all()
                job.stage = "transcript"; job.progress_total = len(docs)
                doc_ids = [d.id for d in docs]
                db.commit()
            for i, doc_id in enumerate(doc_ids):
                with session_scope() as db:
                    job = db.get(Job, job_id); doc = db.get(Document, doc_id)
                    existing = db.execute(select(Chunk.id).where(Chunk.document_id == doc.id)).first()
                    if not existing:
                        text_value = fetch_youtube_transcript(doc.external_id)
                        for ci, piece in enumerate(chunk_text(text_value, settings.chunk_size, settings.chunk_overlap)):
                            db.add(Chunk(document_id=doc.id, chunk_index=ci, text=piece, token_estimate=max(1, len(piece) // 4)))
                    doc.status = "processing"
                    job.progress_current = i + 1
                    job.checkpoint = f"video:{i + 1}/{len(doc_ids)}"
                    db.commit()

        # Profile after transcript expansion so YouTube playlists also get source-level semantics.
        with session_scope() as db:
            job = db.get(Job, job_id); src = db.get(Source, job.source_id)
            self._ensure_source_profile(db, src, ai)
            db.commit()

        # Merge repeated text before paying for embeddings or decisions on it.
        with session_scope() as db:
            job = db.get(Job, job_id); src = db.get(Source, job.source_id)
            job.stage = "dedupe"
            mark_duplicates(db, src)
            db.commit()

        with session_scope() as db:
            job = db.get(Job, job_id); src = db.get(Source, job.source_id)
            chunk_rows = db.execute(
                select(Chunk).join(Document, Chunk.document_id == Document.id)
                .where(Document.source_id == src.id, Chunk.duplicate_of_id.is_(None)).order_by(Chunk.id)
            ).scalars().all()
            job.stage = "index"; job.progress_current = 0; job.progress_total = len(chunk_rows); db.commit()
            chunk_ids = [c.id for c in chunk_rows]
        contextualize_enabled = (settings.contextual_indexing and is_automatic(ai)
                                 and len(chunk_ids) <= settings.contextual_indexing_max_chunks)

        # Batch embedding requests to minimize free-tier API calls.
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
                job.progress_current = min(batch_start + len(batch_ids), len(chunk_ids))
                job.checkpoint = f"embed:{job.progress_current}/{len(chunk_ids)}"
                db.commit()

        # One organize decision per batch: groups named from content plus per-excerpt concepts.
        # A batch commits only after its decision is applied, so a pause or crash replays it intact.
        while True:
            with session_scope() as db:
                job = db.get(Job, job_id); src = db.get(Source, job.source_id)
                own = (Document.source_id == src.id, Chunk.duplicate_of_id.is_(None), Chunk.embedding_status == "done")
                batch = db.execute(
                    select(Chunk).join(Document, Chunk.document_id == Document.id)
                    .where(*own, Chunk.classification_status != "done").order_by(Chunk.id).limit(settings.organize_batch_size)
                ).scalars().all()
                if batch:
                    done, total = self._organize_counts(db, own)
                    size = settings.organize_batch_size
                    part = (done // size + 1, done // size + math.ceil((total - done) / size))
                    organize_batch(db, ai, src, batch, self._source_hint(src), part)
                    db.flush()
                done, total = self._organize_counts(db, own)
                job.stage = "organize"; job.progress_current = done; job.progress_total = total
                job.checkpoint = f"organize:{done}/{total}"
                db.commit()
                if not batch:
                    break

        # Merged duplicates follow their canonical chunk into its group.
        with session_scope() as db:
            job = db.get(Job, job_id); src = db.get(Source, job.source_id)
            duplicates = db.execute(
                select(Chunk).join(Document, Chunk.document_id == Document.id)
                .where(Document.source_id == src.id, Chunk.duplicate_of_id.is_not(None))
            ).scalars().all()
            for chunk in duplicates:
                canonical = db.get(Chunk, chunk.duplicate_of_id)
                if canonical and canonical.group_id:
                    chunk.group_id = canonical.group_id
                    chunk.classification_status = "merged"
            db.commit()

        with session_scope() as db:
            job = db.get(Job, job_id); src = db.get(Source, job.source_id)
            docs = db.execute(select(Document).where(Document.source_id == src.id)).scalars().all()
            for doc in docs:
                doc.status = "done"
            src.status = "done"
            job.status = "done"
            job.stage = "done"
            job.progress_current = job.progress_total
            job.checkpoint = "done"
            db.commit()
