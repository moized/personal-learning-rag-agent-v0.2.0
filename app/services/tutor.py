"""Hand the organized material to an external AI tutor one topic (part) at a time.

Flow: kickoff message (outline + "ask me where to start") -> the user pastes the tutor's
`START: n; SKIP: ...` line back -> `next` returns each topic's verbatim excerpts in order.
Every canonical chunk of the course belongs to exactly one topic, so material is not repeated
and nothing is silently dropped; material no topic claims goes to an "Additional material" topic.
Only a topic that received no excerpts borrows its closest matches by keyword.
"""
import json
import re
from collections import Counter, defaultdict
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..models import Chunk, Course, Document, Evidence, Group, Source, Syllabus, SyllabusTopic
from ..config import settings
from .curriculum import course_chunks_query, syllabus_topics
from .grouping import ConceptIndex, normalize_name
from .ingestion import strip_overlap
from .retrieval import bm25_scores

STATUSES = {"pending", "in_progress", "done", "skipped"}
START_RE = re.compile(r"START\s*[:=#]?\s*(\d+)", re.IGNORECASE)
SKIP_RE = re.compile(r"SKIP\s*[:=]?\s*([^\n;]*)", re.IGNORECASE)


def parse_start_reply(text: str) -> tuple[int | None, list[int]]:
    """Read the tutor's `START: 3; SKIP: 5, 7-8` line (or `SKIP: none`)."""
    match = START_RE.search(text or "")
    start = int(match.group(1)) if match else None
    skip: list[int] = []
    match = SKIP_RE.search(text or "")
    if match:
        for a, b in re.findall(r"(\d+)(?:\s*(?:-|–|to)\s*(\d+))?", match.group(1)):
            skip.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return start, sorted(set(skip))


def _assign_chunks(db: Session, syllabus: Syllabus, topics: list[SyllabusTopic]) -> tuple[dict[int, list[Chunk]], list[Chunk]]:
    """Each chunk goes to the topic claiming most of its concepts, else the topic claiming its group."""
    concepts = ConceptIndex(db)
    groups = {normalize_name(g.name): g.id for g in db.execute(select(Group)).scalars().all()}
    concept_topic: dict[int, SyllabusTopic] = {}
    group_topic: dict[int, SyllabusTopic] = {}
    for topic in topics:
        for name in json.loads(topic.concepts_json or "[]"):
            concept = concepts.find(name)
            if concept is not None:
                concept_topic.setdefault(concept.id, topic)
        for name in json.loads(topic.groups_json or "[]"):
            if normalize_name(name) in groups:
                group_topic.setdefault(groups[normalize_name(name)], topic)

    chunks = db.execute(course_chunks_query(syllabus.course_id)).scalars().all()
    chunk_concepts: dict[int, list[int]] = defaultdict(list)
    if chunks:
        for chunk_id, concept_id in db.execute(
            select(Evidence.chunk_id, Evidence.concept_id).where(Evidence.chunk_id.in_([c.id for c in chunks]))
        ).all():
            chunk_concepts[chunk_id].append(concept_id)

    position = {t.id: t.position for t in topics}
    assigned: dict[int, list[Chunk]] = defaultdict(list)
    leftovers: list[Chunk] = []
    for chunk in chunks:
        votes = Counter(concept_topic[cid].id for cid in chunk_concepts.get(chunk.id, []) if cid in concept_topic)
        if votes:
            winner = max(votes, key=lambda tid: (votes[tid], -position[tid]))
            assigned[winner].append(chunk)
        elif chunk.group_id in group_topic:
            assigned[group_topic[chunk.group_id].id].append(chunk)
        else:
            leftovers.append(chunk)
    return assigned, leftovers


def _material(db: Session, syllabus: Syllabus) -> tuple[list[SyllabusTopic], dict[int, list[Chunk]], set[int]]:
    """Topics to study (with an extra topic for unclaimed material when needed) and their chunks.

    Returns the visible topics, chunks per topic, and the ids of topics filled by a lexical
    fallback because no chunk was assigned to them.
    """
    topics = syllabus_topics(db, syllabus)
    regular = [t for t in topics if t.kind == "topic"]
    assigned, leftovers = _assign_chunks(db, syllabus, regular)
    extra = next((t for t in topics if t.kind == "extra"), None)
    if leftovers and extra is None:
        extra = SyllabusTopic(
            syllabus_id=syllabus.id, position=max((t.position for t in topics), default=0) + 1, kind="extra",
            title="Additional material", purpose="Excerpts that no topic above covers, such as material added after the outline was made.",
            prerequisites_json="[]", concepts_json="[]", groups_json="[]", status="pending", parts_sent=0,
        )
        db.add(extra)
        db.flush()
        topics.append(extra)
    if extra is not None:
        assigned[extra.id] = leftovers

    fallback: set[int] = set()
    all_chunks = None
    for topic in regular:
        if assigned.get(topic.id):
            continue
        # Nothing mapped to this topic: show the closest excerpts by keyword match instead.
        if all_chunks is None:
            all_chunks = db.execute(course_chunks_query(syllabus.course_id)).scalars().all()
        query = " ".join([topic.title, *json.loads(topic.concepts_json or "[]")])
        scores = bm25_scores(query, all_chunks)
        best = [c for c, s in sorted(zip(all_chunks, scores), key=lambda x: x[1], reverse=True)[:6] if s > 0]
        if best:
            assigned[topic.id] = sorted(best, key=lambda c: all_chunks.index(c))
            fallback.add(topic.id)
    visible = [t for t in topics if t.kind == "topic" or assigned.get(t.id)]
    return visible, assigned, fallback


def _blocks(db: Session, chunks: list[Chunk]) -> list[str]:
    """Render excerpts with a provenance header; note where merged duplicates also appeared."""
    also: dict[int, list[str]] = defaultdict(list)
    if chunks:
        rows = db.execute(
            select(Chunk.duplicate_of_id, Source.name).join(Document, Chunk.document_id == Document.id)
            .join(Source, Document.source_id == Source.id).where(Chunk.duplicate_of_id.in_([c.id for c in chunks]))
        ).all()
        for canonical_id, name in rows:
            if name not in also[canonical_id]:
                also[canonical_id].append(name)
    blocks = []
    for chunk in chunks:
        doc = db.get(Document, chunk.document_id)
        src = db.get(Source, doc.source_id) if doc else None
        label = [src.name if src else "source"]
        if doc and doc.title and normalize_name(doc.title) not in normalize_name(label[0]):
            label.append(doc.title)
        if doc and doc.section:
            label.append(doc.section)
        header = f"--- {' › '.join(label)} [chunk {chunk.id}]"
        if also.get(chunk.id):
            header += f" (same text also in: {', '.join(also[chunk.id])})"
        # The overlap repeats the previous chunk, which is sent with this topic or another one.
        prev = db.execute(
            select(Chunk.text).where(Chunk.document_id == chunk.document_id, Chunk.chunk_index == chunk.chunk_index - 1)
        ).scalar_one_or_none()
        text = strip_overlap(prev, chunk.text, settings.chunk_overlap + 40) if prev else chunk.text
        blocks.append(f"{header}\n{text.strip()}")
    return blocks


def _parts(blocks: list[str]) -> list[list[str]]:
    parts: list[list[str]] = [[]]
    size = 0
    for block in blocks:
        if parts[-1] and size + len(block) > settings.study_part_chars:
            parts.append([])
            size = 0
        parts[-1].append(block)
        size += len(block) + 2
    return parts


def _render_part(topic: SyllabusTopic, total: int, number: int, part: int, count: int, blocks: list[str],
                 titles: dict[int, str], fallback: bool) -> str:
    head = f"TOPIC {number} of {total} — {topic.title}" + (f" (part {part} of {count})" if count > 1 else "")
    lines = [head]
    if topic.purpose:
        lines.append(f"Goal: {topic.purpose}")
    prereqs = [n for n in json.loads(topic.prerequisites_json or "[]") if n in titles]
    if prereqs:
        lines.append("Builds on: " + "; ".join(f"topic {n} ({titles[n]})" for n in prereqs))
    concepts = json.loads(topic.concepts_json or "[]")
    if concepts:
        lines.append("Key concepts: " + "; ".join(concepts))
    lines += ["", "SOURCE MATERIAL (verbatim excerpts from my own notes, in reading order):"]
    if fallback:
        lines.append("(No excerpts were filed under this topic, so these are the closest matches.)")
    lines += ["", "\n\n".join(blocks) if blocks else "(No excerpts were found for this topic; teach it from the outline.)", ""]
    if part < count:
        lines.append(f"More material for this topic follows in part {part + 1}. Teach this part from the excerpts above, "
                     f"then ask me to paste part {part + 1}.")
    else:
        lines.append(f"That is all the material for topic {number}. Teach it using the excerpts above as the main source "
                     "(say when you add outside knowledge), then give me 3 short check questions. "
                     "After I answer, ask me to paste the next topic.")
    return "\n".join(lines)


def topic_overview(db: Session, syllabus: Syllabus) -> list[dict]:
    topics, assigned, _ = _material(db, syllabus)
    out = []
    for number, topic in enumerate(topics, 1):
        chunks = assigned.get(topic.id, [])
        out.append({
            "id": topic.id, "number": number, "kind": topic.kind, "title": topic.title, "purpose": topic.purpose,
            "prerequisites": json.loads(topic.prerequisites_json or "[]"),
            "concepts": json.loads(topic.concepts_json or "[]"), "groups": json.loads(topic.groups_json or "[]"),
            "status": topic.status, "parts_sent": topic.parts_sent,
            "parts_total": len(_parts(_blocks(db, chunks))), "excerpts": len(chunks),
        })
    return out


def kickoff_message(db: Session, syllabus: Syllabus) -> str:
    topics, _, _ = _material(db, syllabus)
    course = db.get(Course, syllabus.course_id) if syllabus.course_id else None
    name = course.name if course else "my study material"
    lines = [
        f"You are going to be my tutor for \"{name}\". A local tool has organized my own learning material "
        "(lectures, books, transcripts) into the outline below. I will paste the material one topic at a time; "
        "treat what I paste as the main source of truth.",
    ]
    if course and course.description:
        lines.append(f"About the course: {course.description}")
    lines += ["", "STUDY OUTLINE"]
    for number, topic in enumerate(topics, 1):
        line = f"{number}. {topic.title}"
        if topic.purpose:
            line += f" — {topic.purpose}"
        prereqs = json.loads(topic.prerequisites_json or "[]")
        if prereqs:
            line += f" (builds on {', '.join(map(str, prereqs))})"
        if topic.status in {"done", "skipped"}:
            line += f" [{'already covered' if topic.status == 'done' else 'skipped'}]"
        lines.append(line)
    lines += [
        "",
        "How we will work:",
        "1. Do not start teaching yet. First ask me where I want to start and whether I want to skip any topics "
        "(for example, ones I already know). You may suggest a starting point based on the prerequisites.",
        "2. Once I have decided, write one line exactly in this form so I can copy it back into my tool:",
        "   START: 3; SKIP: 5, 7",
        "   (use \"SKIP: none\" when nothing is skipped).",
        "3. I will then paste one topic at a time, labelled \"TOPIC n of N — title\" (long topics come in parts). "
        "Teach each one from the pasted material, connect it to earlier topics, and point out anything the material leaves unclear.",
        "4. At the end of each topic, give me 3 short check questions, then ask me to paste the next one.",
    ]
    return "\n".join(lines)


def set_topic_status(db: Session, topic: SyllabusTopic, status: str) -> None:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}")
    topic.status = status
    if status in {"pending", "skipped"}:
        topic.parts_sent = 0


def start_session(db: Session, syllabus: Syllabus, start: int | None, skip: list[int]) -> None:
    """Apply the user's choice: topics before `start` and those in `skip` are skipped; the rest are queued."""
    topics, _, _ = _material(db, syllabus)
    if start is not None and not 1 <= start <= len(topics):
        raise ValueError(f"START must be between 1 and {len(topics)}")
    skip_set = set(skip)
    for number, topic in enumerate(topics, 1):
        if topic.status == "done":
            continue
        if (start is not None and number < start) or number in skip_set:
            set_topic_status(db, topic, "skipped")
        elif topic.status == "skipped":
            set_topic_status(db, topic, "pending")


def part_message(db: Session, syllabus: Syllabus, topic_id: int, part: int) -> dict:
    """Render one part of a topic without changing any progress (for re-sending)."""
    topics, assigned, fallback = _material(db, syllabus)
    titles = {n: t.title for n, t in enumerate(topics, 1)}
    for number, topic in enumerate(topics, 1):
        if topic.id != topic_id:
            continue
        parts = _parts(_blocks(db, assigned.get(topic.id, [])))
        if not 1 <= part <= len(parts):
            raise LookupError(f"Topic {number} has {len(parts)} part(s)")
        text = _render_part(topic, len(topics), number, part, len(parts), parts[part - 1], titles, topic.id in fallback)
        return {"topic_id": topic.id, "number": number, "title": topic.title, "part": part, "parts": len(parts), "text": text}
    raise LookupError("Topic not found in this syllabus")


def next_message(db: Session, syllabus: Syllabus) -> dict:
    """Return the next part to paste and record it as sent."""
    topics, assigned, fallback = _material(db, syllabus)
    titles = {n: t.title for n, t in enumerate(topics, 1)}
    while True:
        current = next((t for t in topics if t.status == "in_progress"), None) or \
            next((t for t in topics if t.status == "pending"), None)
        if current is None:
            return {"done": True, "text": None, "message": "Every topic is done or skipped."}
        number = topics.index(current) + 1
        parts = _parts(_blocks(db, assigned.get(current.id, [])))
        if current.parts_sent >= len(parts):
            current.status = "done"
            continue
        part = current.parts_sent + 1
        text = _render_part(current, len(topics), number, part, len(parts), parts[part - 1], titles, current.id in fallback)
        current.parts_sent = part
        current.status = "done" if part == len(parts) else "in_progress"
        remaining = sum(1 for t in topics if t.status in {"pending", "in_progress"})
        return {"done": False, "topic_id": current.id, "number": number, "title": current.title,
                "part": part, "parts": len(parts), "remaining_topics": remaining, "text": text}
