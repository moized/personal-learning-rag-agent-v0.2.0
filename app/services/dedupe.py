"""Merge chunks that repeat text already stored (re-uploads, repeated slides, recap segments).

A new chunk is a duplicate when nearly all of its word 3-gram shingles already occur in one
earlier chunk of the same course (coverage >= `duplicate_threshold`). Shingles are taken from
each chunk's own text, without the overlap the chunker copies from the previous chunk, so a
repeated paragraph still matches at a different position. Scoping to one course keeps
course-filtered retrieval complete. The earliest chunk stays canonical; later copies point at
it through `duplicate_of_id` and are skipped by embedding, grouping, retrieval and tutoring.
Paraphrases are not merged here: they share a group and concepts instead.
"""
import re
import zlib
from collections import Counter, defaultdict
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..models import Chunk, Document, Source
from ..config import settings
from .ingestion import strip_overlap

WORD_RE = re.compile(r"\w+", re.UNICODE)


def shingles(text: str, k: int = 3) -> set[int]:
    words = [w.lower() for w in WORD_RE.findall(text or "")]
    if not words:
        return set()
    if len(words) < k:
        return {zlib.crc32(" ".join(words).encode("utf-8"))}
    return {zlib.crc32(" ".join(words[i:i + k]).encode("utf-8")) for i in range(len(words) - k + 1)}


def _course_filter(course_id: int | None):
    return Source.course_id == course_id if course_id is not None else Source.course_id.is_(None)


def mark_duplicates(db: Session, src: Source, threshold: float | None = None) -> int:
    """Mark this source's near-duplicate chunks. Resume-safe: processed chunks are skipped."""
    threshold = settings.duplicate_threshold if threshold is None else threshold
    rows = db.execute(
        select(Chunk, Document.source_id).join(Document, Chunk.document_id == Document.id)
        .join(Source, Document.source_id == Source.id).where(_course_filter(src.course_id))
    ).all()
    pending = sorted((c for c, source_id in rows if source_id == src.id and c.dedupe_status != "done"), key=lambda c: c.id)
    if not pending:
        return 0
    texts = {(c.document_id, c.chunk_index): c.text for c, _ in rows}

    def own_shingles(chunk: Chunk) -> set[int]:
        prev = texts.get((chunk.document_id, chunk.chunk_index - 1))
        return shingles(strip_overlap(prev, chunk.text, settings.chunk_overlap + 40) if prev else chunk.text)

    index: defaultdict[int, list[int]] = defaultdict(list)
    for chunk, _ in rows:
        if chunk.dedupe_status == "done" and chunk.duplicate_of_id is None:
            for s in own_shingles(chunk):
                index[s].append(chunk.id)

    merged = 0
    for chunk in pending:
        sh = own_shingles(chunk)
        overlap: Counter[int] = Counter()
        for s in sh:
            overlap.update(index.get(s, ()))
        best = max(overlap.items(), key=lambda item: (item[1], -item[0]), default=None)
        if best is not None and best[1] / len(sh) >= threshold:
            chunk.duplicate_of_id = best[0]
            merged += 1
        else:
            for s in sh:
                index[s].append(chunk.id)
        chunk.dedupe_status = "done"
    return merged
