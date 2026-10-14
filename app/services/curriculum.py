import json
from collections import defaultdict
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..models import Group, Concept, Evidence, Chunk, Document, Source, Course, Syllabus, SyllabusTopic
from ..providers.base import AIProvider, is_automatic
from ..config import settings
from .grouping import merge_concepts, normalize_name, LEGACY_GROUP_DESCRIPTION
from .retrieval import retrieve, group_candidates

SYLLABUS_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "topic title"},
                    "purpose": {"type": "string", "description": "what the student will be able to do"},
                    "prerequisites": {"type": "array", "items": {"type": "integer"}},
                    "groups": {"type": "array", "items": {"type": "string", "description": "exact group name"}},
                    "concepts": {"type": "array", "items": {"type": "string", "description": "exact concept name"}},
                },
                "required": ["title", "purpose", "groups", "concepts"],
            },
        },
        "duplicate_concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keep": {"type": "string", "description": "concept name to keep"},
                    "merge": {"type": "array", "items": {"type": "string", "description": "concept name that means the same"}},
                },
                "required": ["keep", "merge"],
            },
        },
    },
    "required": ["topics"],
}


def course_chunks_query(course_id: int | None):
    """Canonical (non-duplicate), organized chunks of a course, in reading order."""
    stmt = (
        select(Chunk).join(Document, Chunk.document_id == Document.id).join(Source, Document.source_id == Source.id)
        .where(Chunk.duplicate_of_id.is_(None), Chunk.classification_status == "done", Chunk.embedding_status == "done")
    )
    if course_id is not None:
        stmt = stmt.where(Source.course_id == course_id)
    return stmt.order_by(Source.id, Document.id, Chunk.chunk_index)


def _course_groups(db: Session, course_id: int | None) -> list[tuple[Group, list[str]]]:
    """Groups holding this course's material, in order of first appearance, with their concepts."""
    chunks = db.execute(course_chunks_query(course_id)).scalars().all()
    order: dict[int, int] = {}
    for i, chunk in enumerate(chunks):
        if chunk.group_id is not None:
            order.setdefault(chunk.group_id, i)
    concepts: dict[int, set[str]] = defaultdict(set)
    chunk_groups = {c.id: c.group_id for c in chunks}
    if chunk_groups:
        rows = db.execute(
            select(Evidence.chunk_id, Concept.canonical_name).join(Concept, Evidence.concept_id == Concept.id)
            .where(Evidence.chunk_id.in_(list(chunk_groups)))
        ).all()
        for chunk_id, name in rows:
            if chunk_groups.get(chunk_id) is not None:
                concepts[chunk_groups[chunk_id]].add(name)
    groups = [db.get(Group, gid) for gid in sorted(order, key=order.get)]
    return [(g, sorted(concepts.get(g.id, set()))) for g in groups if g is not None]


def build_syllabus_prompt(course: Course | None, groups: list[tuple[Group, list[str]]]) -> str:
    blocks = []
    for group, concepts in groups:
        desc = group.description if group.description and group.description != LEGACY_GROUP_DESCRIPTION else ""
        blocks.append(f"## {group.name}" + (f" — {desc}" if desc else "") + "\n"
                      + ("concepts: " + "; ".join(concepts[:80]) if concepts else "concepts: (none extracted)"))
    return "\n".join([
        "Design a study outline (syllabus) from a student's organized learning material.",
        f"Course: {course.name if course else 'all material'}",
        f"Course description: {(course.description if course else None) or 'none'}",
        f"Student's preferred outline (a preference only, not evidence that a topic exists): {(course.outline if course else None) or 'none'}",
        "",
        "Rules:",
        "- Order topics so that prerequisites come before the topics that depend on them. Topic 1 is the first item.",
        "- Make each topic small enough for one study session; merge tiny groups and split very broad ones.",
        "- Every group below must be covered: list the exact group names each topic draws on in `groups`.",
        "- List the exact concept names each topic covers in `concepts`. Put each concept in only one topic.",
        "- `prerequisites` holds the numbers of earlier topics this one builds on.",
        "- Do not invent topics that the material does not cover.",
        "- If two concept names below mean the same thing (synonym, abbreviation, spelling variant), report them in "
        "`duplicate_concepts` with the name to keep.",
        "",
        "MATERIAL GROUPS:",
        *blocks,
    ])


def _names(item: dict, key: str) -> list[str]:
    value = item.get(key)
    return [str(v).strip() for v in value if str(v).strip()] if isinstance(value, list) else []


def _clean_topics(raw) -> list[dict]:
    topics = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or not str(item.get("title") or "").strip():
            continue
        prereqs = []
        for value in item.get("prerequisites") or []:
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= number <= len(topics):
                prereqs.append(number)
        topics.append({
            "title": str(item["title"]).strip()[:500],
            "purpose": str(item.get("purpose") or "").strip() or None,
            "prerequisites": sorted(set(prereqs)),
            "groups": _names(item, "groups"),
            "concepts": _names(item, "concepts"),
        })
    return topics


def _fallback_topics(groups: list[tuple[Group, list[str]]]) -> list[dict]:
    """No model answer (offline mode): one topic per group, in the order the material appeared."""
    return [{
        "title": group.name,
        "purpose": group.description if group.description != LEGACY_GROUP_DESCRIPTION else None,
        "prerequisites": [],
        "groups": [group.name],
        "concepts": concepts,
    } for group, concepts in groups]


def get_syllabus(db: Session, course_id: int | None) -> Syllabus | None:
    cond = Syllabus.course_id == course_id if course_id is not None else Syllabus.course_id.is_(None)
    return db.execute(select(Syllabus).where(cond).order_by(Syllabus.id.desc())).scalars().first()


def syllabus_topics(db: Session, syllabus: Syllabus) -> list[SyllabusTopic]:
    return db.execute(
        select(SyllabusTopic).where(SyllabusTopic.syllabus_id == syllabus.id).order_by(SyllabusTopic.position)
    ).scalars().all()


def render_syllabus(topics: list[SyllabusTopic]) -> str:
    lines = []
    for topic in topics:
        line = f"{topic.position}. **{topic.title}**"
        if topic.purpose:
            line += f" — {topic.purpose}"
        lines.append(line)
        details = []
        prereqs = json.loads(topic.prerequisites_json or "[]")
        if prereqs:
            details.append("Builds on: " + ", ".join(map(str, prereqs)))
        concepts = json.loads(topic.concepts_json or "[]")
        if concepts:
            details.append("Concepts: " + "; ".join(concepts))
        if details:
            lines.append("   " + " · ".join(details))
    return "\n".join(lines)


def generate_syllabus(db: Session, ai: AIProvider, course_id: int | None = None) -> Syllabus | None:
    """Ask for an ordered outline, apply its concept merges, and store it with per-topic status.

    Regenerating keeps the status of topics whose title is unchanged.
    """
    course = db.get(Course, course_id) if course_id else None
    groups = _course_groups(db, course_id)
    if not groups:
        return None
    result = ai.generate_json(
        build_syllabus_prompt(course, groups), task="syllabus", schema=SYLLABUS_SCHEMA,
        label=f"Plan the study outline for {course.name if course else 'all your material'}",
    )
    result = result if isinstance(result, dict) else {}
    for item in result.get("duplicate_concepts") or []:
        if isinstance(item, dict) and isinstance(item.get("merge"), list):
            merge_concepts(db, str(item.get("keep") or ""), [str(m) for m in item["merge"]])
    topics = _clean_topics(result.get("topics")) or _fallback_topics(groups)

    previous_status: dict[str, tuple[str, int]] = {}
    old = get_syllabus(db, course_id)
    if old is not None:
        for topic in syllabus_topics(db, old):
            if topic.kind == "topic":
                previous_status[normalize_name(topic.title)] = (topic.status, topic.parts_sent)
            db.delete(topic)
        db.flush()
        db.delete(old)
        db.flush()

    syllabus = Syllabus(course_id=course_id, text="")
    db.add(syllabus)
    db.flush()
    rows = []
    for position, item in enumerate(topics, 1):
        status, parts_sent = previous_status.get(normalize_name(item["title"]), ("pending", 0))
        row = SyllabusTopic(
            syllabus_id=syllabus.id, position=position, kind="topic", title=item["title"], purpose=item["purpose"],
            prerequisites_json=json.dumps(item["prerequisites"]),
            concepts_json=json.dumps(item["concepts"], ensure_ascii=False),
            groups_json=json.dumps(item["groups"], ensure_ascii=False),
            status=status, parts_sent=parts_sent,
        )
        db.add(row)
        rows.append(row)
    db.flush()
    syllabus.text = render_syllabus(rows)
    return syllabus


def build_study_pack_prompt(db: Session, ai: AIProvider, course_id: int | None, topic: str, top_k: int = 12) -> tuple[str, list[dict]]:
    q = ai.embed([topic])[0]
    candidate_groups = group_candidates(db, q, None, limit=settings.retrieval_group_limit, course_id=course_id)
    scored = retrieve(db, ai, topic, candidate_groups, top_k=top_k, course_id=course_id)
    if len(scored) < max(2, top_k // 2):
        fallback = retrieve(db, ai, topic, None, top_k=top_k, course_id=course_id)
        seen = {c.id for c, _ in scored}
        scored.extend((c, s) for c, s in fallback if c.id not in seen)
        scored = scored[:top_k]
    if not scored:
        return "No relevant material was found for this topic in the selected course.", []

    evidence: list[str] = []
    refs: list[dict] = []
    for c, score in scored:
        doc = db.get(Document, c.document_id)
        src = db.get(Source, doc.source_id) if doc else None
        label = f"{src.name if src else 'source'} / {doc.title if doc else 'document'} / {doc.section or 'section'}"
        evidence.append(f"[{label} | chunk:{c.id}]\n{c.text}")
        refs.append({"chunk_id": c.id, "score": score, "source": src.name if src else None, "section": doc.section if doc else None})

    course = db.get(Course, course_id) if course_id else None
    prompt = f"""Prepare a rigorous study pack for a student who may paste it into another AI tutor.
Topic: {topic}
Course: {course.name if course else 'all material'}
Use only the supplied evidence. Synthesize repeated explanations instead of repeating them. Preserve disagreements or uncertainty.
Include source labels and [chunk:ID] citations next to factual claims where appropriate.
Structure:
1. Learning objective
2. Prerequisites
3. Core explanation
4. Key concepts and terminology
5. Connections and mental model
6. Worked/intuitive example only when supported
7. Common confusions and failure modes
8. Five self-check questions
9. Suggested next topic

Evidence:
{chr(10).join(evidence)}"""
    if not is_automatic(ai):
        # Without an API the prompt itself, evidence included, is what the user pastes into their AI.
        return prompt, refs
    return ai.generate_text(prompt, task="study"), refs
