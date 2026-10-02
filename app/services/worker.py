import json
import re
import threading
import time
from pathlib import Path
from sqlalchemy import select
from ..config import settings
from ..db import session_scope
from ..models import Source, Document, Chunk, Job, Group
from ..providers.factory import provider
from .ingestion import parse_file, chunk_text, list_youtube_videos, fetch_youtube_transcript
from .grouping import choose_group, extract_group_concepts, merge_centroid
from .knowledge_graph import add_sequence_relations, extract_group_relations
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
        while not self.stop_event.is_set():
            with session_scope() as db:
                from .queue import claim_next
                job = claim_next(db)
            if not job:
                time.sleep(settings.poll_seconds)
                continue
            try:
                self.process_job(job.id)
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
        videos = list_youtube_videos(src.uri)
        for video_id, title in videos:
            doc = db.execute(
                select(Document).where(
                    Document.source_id == src.id,
                    Document.external_id == video_id,
                )
            ).scalar_one_or_none()
            if not doc:
                doc = Document(
                    source_id=src.id,
                    external_id=video_id,
                    title=title,
                    section="youtube transcript",
                    status="pending",
                )
                db.add(doc)
        return len(videos)

    def _ensure_source_profile(self, db, src: Source, ai):
        if src.profile_json:
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
            src.profile_json = json.dumps(
                ai.generate_json(profile_prompt, task="classification", schema=schema),
                ensure_ascii=False,
            )
        except Exception:
            src.profile_json = json.dumps(
                {"domain": src.domain_hint, "topics": [src.topic_hint] if src.topic_hint else []},
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
                    if not existing:
                        text_value = fetch_youtube_transcript(doc.external_id)
                        for ci, piece in enumerate(
                            chunk_text(text_value, settings.chunk_size, settings.chunk_overlap)
                        ):
                            db.add(
                                Chunk(
                                    document_id=doc.id,
                                    chunk_index=ci,
                                    chunk_kind="leaf",
                                    text=piece,
                                    token_estimate=max(1, len(piece) // 4),
                                )
                            )
                    db.flush()
                    self._ensure_chunk_hierarchy(db, doc)
                    doc.status = "processing"
                    job.progress_current = i + 1
                    job.checkpoint = f"video:{i + 1}/{len(doc_ids)}"
                    db.commit()

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
            job.stage = "index"
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
                job.stage = "organize"
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
                    extract_group_relations(db, ai, group.id)
                    db.query(Chunk).filter(
                        Chunk.group_id == group.id,
                        Chunk.chunk_kind == "leaf",
                    ).update({"cluster_status": "done"}, synchronize_session=False)
                job.progress_current = i
                job.checkpoint = f"group:{i}/{len(group_ids)}"
                db.commit()

        with session_scope() as db:
            job = db.get(Job, job_id)
            src = db.get(Source, job.source_id)
            docs = db.execute(
                select(Document).where(Document.source_id == src.id)
            ).scalars().all()
            for doc in docs:
                doc.status = "done"
            src.status = "done"
            job.status = "done"
            job.stage = "done"
            job.progress_current = job.progress_total
            job.checkpoint = "done"
            db.commit()
