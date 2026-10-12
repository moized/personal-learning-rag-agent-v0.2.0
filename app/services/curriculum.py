import json
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..models import Group, Concept, Evidence, Chunk, Document, Source, Course
from ..providers.base import AIProvider
from ..config import settings
from .retrieval import retrieve, group_candidates
from .learning_paths import build_syllabus_structured as build_source_first_syllabus
from .spaces import in_space


def _course_context(db: Session, course_id: int | None) -> tuple[Course | None, list[tuple[str, list[str]]]]:
    course = db.get(Course, course_id) if course_id else None
    stmt = select(Chunk).join(Document, Chunk.document_id == Document.id).join(Source, Document.source_id == Source.id).where(Chunk.embedding_status == "done", Chunk.chunk_kind == "leaf", Source.status == "done")
    if course_id:
        stmt = stmt.where(in_space(course_id))
    chunk_ids = {c.id for c in db.execute(stmt).scalars().all()}
    scoped_groups = set(db.execute(
        select(Chunk.group_id).where(Chunk.id.in_(chunk_ids), Chunk.group_id.is_not(None)).distinct()
    ).scalars().all())
    rows = db.execute(select(Group, Concept).join(Concept, Concept.group_id == Group.id)).all()
    grouped: dict[str, set[str]] = {}
    for group, concept in rows:
        # A group is shared by semantically similar chunks across sources. Evidence
        # links are a small sample, so requiring a direct link to this source hid
        # every concept when a newly added source joined an existing group.
        if group.id in scoped_groups:
            grouped.setdefault(group.name, set()).add(concept.canonical_name)
    return course, [(name, sorted(values)) for name, values in sorted(grouped.items())]


def _legacy_group_syllabus_structured(db: Session, ai: AIProvider, course_id: int | None = None) -> tuple[str, list[dict]]:
    course, grouped = _course_context(db, course_id)
    if not grouped:
        return "No organized learning material is available for this scope yet.", []
    lines = [f"## {name}\n" + ", ".join(concepts[:80]) for name, concepts in grouped]
    course_desc = course.description if course else None
    saved_outline = course.outline if course else None
    prompt = f"""Create an updated hierarchical learning syllabus from the currently indexed source concepts below.
Rebuild and merge the outline from these concepts when the indexed content revision changes. The saved outline is only a structural preference; do not return it unchanged when the indexed concepts differ.
Order prerequisites before dependent topics, while keeping independent or parallel approaches as siblings. Do not invent topics absent from the organized concepts.
Split broad topics into an outline node with focused child topics. Mark nodes with children as type `outline`; mark teachable leaf topics as type `study`. Keep stable IDs such as T1, T1.1, T1.2. A topic with type `outline` is for navigation and must not be treated as a study-pack request.
Return structured JSON with a `nodes` array. Each node has `id`, `parent_id` (use the string `none` for roots), `type` (outline or study), `title`, `purpose`, `prerequisites`, `parallel`, and `source_groups` (string array).
Course description: {course_desc or 'none'}
Saved outline preference: {saved_outline or 'none'}
Current indexed concepts:
{chr(10).join(lines)}"""
    schema = {"type": "object", "properties": {"nodes": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"}, "parent_id": {"type": "string"},
        "type": {"type": "string", "enum": ["outline", "study"]}, "title": {"type": "string"},
        "purpose": {"type": "string"}, "prerequisites": {"type": "string"},
        "parallel": {"type": "string"}, "source_groups": {"type": "array", "items": {"type": "string"}},
    }, "required": ["id", "parent_id", "type", "title", "purpose", "prerequisites", "parallel", "source_groups"]}}}, "required": ["nodes"]}
    result = ai.generate_json(prompt, task="syllabus", schema=schema)
    nodes = []
    seen_ids: set[str] = set()
    for raw in result.get("nodes", []):
        node_id = str(raw.get("id", "")).strip()
        title = str(raw.get("title", "")).strip()
        node_type = raw.get("type")
        if not node_id or node_id in seen_ids or not title or node_type not in {"outline", "study"}:
            continue
        seen_ids.add(node_id)
        parent = str(raw.get("parent_id") or "none").strip()
        nodes.append({"node_id": node_id, "parent_node_id": None if parent.casefold() in {"", "none", "root"} else parent, "node_type": node_type,
                      "title": title, "purpose": str(raw.get("purpose", "")),
                      "prerequisites": str(raw.get("prerequisites", "")), "parallel": str(raw.get("parallel", "")),
                      "source_groups": [str(x) for x in raw.get("source_groups", [])]})
    ids = {node["node_id"] for node in nodes}
    nodes = [node for node in nodes if node["parent_node_id"] is None or node["parent_node_id"] in ids]
    if not nodes:
        return "No valid outline topics could be generated from the indexed material.", []
    rows = ["| # | ID | Parent ID | Topic | Type | Purpose | Prerequisites | Parallel approaches | Source groups |", "|---|---|---|---|---|---|---|---|---|"]
    for index, node in enumerate(nodes, 1):
        values = [str(index), node["node_id"], node["parent_node_id"] or "none", node["title"], node["node_type"], node["purpose"], node["prerequisites"], node["parallel"], ", ".join(node["source_groups"])]
        rows.append("| " + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in values) + " |")
    persisted = [{"node_id": node["node_id"], "parent_node_id": node["parent_node_id"], "node_type": node["node_type"], "title": node["title"], "metadata_json": json.dumps({key: node[key] for key in ("purpose", "prerequisites", "parallel", "source_groups")}, ensure_ascii=False)} for node in nodes]
    return "\n".join(rows), persisted


def build_syllabus_structured(db: Session, ai: AIProvider, course_id: int | None = None, **options) -> tuple[str, list[dict]]:
    return build_source_first_syllabus(db, ai, course_id, **options)


def build_syllabus(db: Session, ai: AIProvider, course_id: int | None = None) -> str:
    return build_syllabus_structured(db, ai, course_id)[0]


def build_study_pack_prompt(db: Session, ai: AIProvider, course_id: int | None, topic: str, top_k: int = 12, retrieval_scope: str = "space_only") -> tuple[str, list[dict]]:
    q = ai.embed([topic])[0]
    candidate_groups = group_candidates(db, q, None, limit=settings.retrieval_group_limit, course_id=course_id)
    scored = retrieve(db, ai, topic, candidate_groups, top_k=top_k, course_id=course_id)
    supplemental_ids: set[int] = set()
    if retrieval_scope == "space_and_related" and course_id is not None:
        fallback = retrieve(db, ai, topic, None, top_k=top_k, course_id=None)
        seen = {c.id for c, _ in scored}
        floor = scored[0][1] * .5 if scored else .01
        additions = [(c, s) for c, s in fallback if c.id not in seen and s >= floor]
        supplemental_ids = {c.id for c, _ in additions}
        scored.extend(additions)
        scored.sort(key=lambda item: item[1], reverse=True)
        scored = scored[:top_k]
        supplemental_ids.intersection_update(c.id for c, _ in scored)
    if not scored:
        return "No relevant material was found for this topic in the selected course.", []

    evidence: list[str] = []
    refs: list[dict] = []
    for c, score in scored:
        doc = db.get(Document, c.document_id)
        src = db.get(Source, doc.source_id) if doc else None
        label = f"{src.name if src else 'source'} / {doc.title if doc else 'document'} / {doc.section or 'section'}"
        evidence.append(f"[{label} | chunk:{c.id}]\n{c.text}")
        course = db.get(Course, src.course_id) if src and src.course_id else None
        refs.append({"chunk_id": c.id, "score": score, "source": src.name if src else None, "section": doc.section if doc else None, "course_id": src.course_id if src else None, "course": course.name if course else None, "is_supplemental": c.id in supplemental_ids})

    course = db.get(Course, course_id) if course_id else None
    prompt = f"""Prepare a rigorous study pack for a student who may paste it into another AI tutor.
Topic: {topic}
Course: {course.name if course else 'all material'}
Use only the supplied evidence. Synthesize repeated explanations instead of repeating them. Preserve disagreements or uncertainty.
Include source labels and [chunk:ID] citations next to factual claims where appropriate.
Structure:
1. Learning objective
2. Prerequisites: list a prerequisite only when the supplied evidence says it is required; do not turn optional preprocessing or alternative approaches into dependencies. If none is established, say so.
3. Core explanation
4. Key concepts and terminology
5. Connections and mental model
6. Worked/intuitive example only when supported
7. Common confusions and failure modes
8. Five self-check questions
9. Suggested next topic

Evidence:
{chr(10).join(evidence)}"""
    return ai.generate_text(prompt, task="study"), refs


def build_external_study_pack(
    db: Session,
    ai: AIProvider,
    course_id: int | None,
    topic: str,
    top_k: int = 20,
    tutor: str = "general",
    max_input_tokens: int = 6000,
    retrieval_scope: str = "space_only",
) -> tuple[str, list[dict], dict]:
    """Build a source-grounded handoff pack without asking an LLM to rewrite it."""
    pack_limit = min(max(top_k, 1), 20)
    scored = retrieve(db, ai, topic, top_k=pack_limit, course_id=course_id, use_reranker=False)
    supplemental_ids: set[int] = set()
    if retrieval_scope == "space_and_related" and course_id is not None:
        supplemental = retrieve(db, ai, topic, top_k=pack_limit, course_id=None, use_reranker=False)
        seen = {chunk.id for chunk, _score in scored}
        floor = scored[0][1] * .5 if scored else .01
        additions = [(chunk, score) for chunk, score in supplemental if chunk.id not in seen and score >= floor]
        supplemental_ids = {chunk.id for chunk, _ in additions}
        scored.extend(additions)
        scored.sort(key=lambda item: item[1], reverse=True)
        scored = scored[:pack_limit]
        supplemental_ids.intersection_update(chunk.id for chunk, _ in scored)
    if not scored:
        return "No relevant material was found for this topic in the selected course.", [], {"estimated_tokens": 0, "omitted_passages": 0, "suggested_subtopics": []}

    course = db.get(Course, course_id) if course_id else None
    tutor_names = {"chatgpt": "ChatGPT", "gemini": "Gemini", "claude": "Claude"}
    tutor_name = tutor_names.get(tutor.lower(), "any AI tutor")
    excerpts: list[str] = []
    refs: list[dict] = []
    omitted_ids: set[int] = set()
    used_token_budget = 0
    # The external model is selected by the user. Keep reserve for its system prompt,
    # conversation and answer, and use a conservative multilingual text estimate.
    source_budget = max(500, max_input_tokens - 900)
    for chunk, _score in scored:
        doc = db.get(Document, chunk.document_id)
        source = db.get(Source, doc.source_id) if doc else None
        label = source.name if source else "Unknown source"
        section = doc.section if doc else None
        title = doc.title if doc else None
        citation = " / ".join(part for part in (label, title, section, f"chunk:{chunk.id}") if part)
        excerpt = f"### {citation}\n{chunk.text.strip()}"
        excerpt_tokens = (len(excerpt) + 2) // 3
        if used_token_budget + excerpt_tokens > source_budget:
            omitted_ids.add(chunk.id)
            continue
        excerpts.append(excerpt)
        used_token_budget += excerpt_tokens
        course_ref = db.get(Course, source.course_id) if source and source.course_id else None
        refs.append({"chunk_id": chunk.id, "source": label, "title": title, "section": section, "course_id": source.course_id if source else None, "course": course_ref.name if course_ref else None, "is_supplemental": chunk.id in supplemental_ids})

    if not excerpts:
        return "No source passage fits this pack size. Increase the source limit or choose a narrower topic.", [], {"estimated_tokens": 0, "omitted_passages": len(scored), "suggested_subtopics": []}

    instruction = f"""You are my patient, rigorous study tutor. I am studying: {topic}.
Course: {course.name if course else 'My learning materials'}.
Use the source excerpts below as the primary evidence. Explain clearly, define terms, connect ideas, and distinguish what the sources say from your own added context. Cite claims with the source labels provided. If the excerpts do not support an answer, say what is missing instead of guessing.
Start by giving me a short overview, then teach one idea at a time. Ask a quick check question and wait for my answer before moving on. Adapt to my answers and correct misunderstandings kindly. Do not dump the whole lesson at once.
This prompt is prepared for {tutor_name}; keep the response interactive and easy to read."""
    pack = f"""# Study pack: {topic}

## Tutor instructions (paste this with the excerpts)
{instruction}

## Source excerpts
{chr(10).join(excerpts)}
"""
    estimated_tokens = (len(pack) + 2) // 3
    suggested_subtopics: list[str] = []
    if omitted_ids:
        rows = db.execute(
            select(Concept.canonical_name)
            .join(Evidence, Evidence.concept_id == Concept.id)
            .where(Evidence.chunk_id.in_(omitted_ids))
            .distinct()
        ).all()
        topic_key = topic.casefold().strip()
        suggested_subtopics = [name for (name,) in rows if name.casefold().strip() != topic_key][:8]
    meta = {
        "estimated_tokens": estimated_tokens,
        "max_input_tokens": max_input_tokens,
        "omitted_passages": len(omitted_ids),
        "suggested_subtopics": suggested_subtopics,
        "supplemental_source_count": len(supplemental_ids),
    }
    return pack, refs, meta
