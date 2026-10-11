"""Source-first course identity and curriculum synthesis."""
import hashlib
import itertools
import json
import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Chunk, Concept, Document, Evidence, Section, Source, SourceRelationship
from ..providers.base import AIProvider, ProviderQuotaExceededError
from .grouping import cosine

RELATION_TYPES = {
    "SAME_COURSE", "SAME_LEARNING_PATH", "COMPLEMENTARY_RESOURCE",
    "SAME_DOMAIN_DIFFERENT_COURSE", "RELATED_BUT_SEPARATE", "UNRELATED",
}
MERGE_TYPES = {"SAME_COURSE", "SAME_LEARNING_PATH", "COMPLEMENTARY_RESOURCE"}
_UNCLASSIFIED_RATIONALE = "Not enough evidence to confirm a shared learning path."


def _is_unclassified_default(row: SourceRelationship) -> bool:
    """A guess saved when the AI call failed or skipped a pair; such rows are always asked again."""
    return row.confidence == 0.2 and row.rationale == _UNCLASSIFIED_RATIONALE


def _profile(source: Source) -> dict:
    try:
        value = json.loads(source.profile_json or "{}")
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def group_source_paths(source_ids: list[int], relationships: list[dict]) -> list[list[int]]:
    """Merge only identity-confirmed and complementary sources into one path."""
    parent = {item: item for item in source_ids}

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for relation in relationships:
        if relation.get("relation_type", relation.get("type")) not in MERGE_TYPES:
            continue
        left = relation.get("source_a_id", relation.get("source_a"))
        right = relation.get("source_b_id", relation.get("source_b"))
        if left in parent and right in parent:
            parent[find(right)] = find(left)
    result: dict[int, list[int]] = {}
    for item in source_ids:
        result.setdefault(find(item), []).append(item)
    return sorted((sorted(items) for items in result.values()), key=lambda items: items[0])


def _section_rows(db: Session, source: Source) -> list[dict]:
    """A source's sections in reading order, each with its main concepts and a concept-dense excerpt.

    Sources processed before sections existed fall back to one pseudo-section built from
    their concepts and first passage, so they still appear in plans until they are reorganized.
    """
    sections = db.execute(select(Section).where(Section.source_id == source.id).order_by(Section.position)).scalars().all()
    if not sections:
        concepts = db.execute(
            select(Concept.canonical_name).join(Evidence, Evidence.concept_id == Concept.id)
            .join(Chunk, Evidence.chunk_id == Chunk.id).join(Document, Chunk.document_id == Document.id)
            .where(Document.source_id == source.id, Chunk.chunk_kind == "leaf")
            .group_by(Concept.id).order_by(func.count(Evidence.id).desc(), Concept.canonical_name).limit(60)
        ).scalars().all()
        first = db.execute(
            select(Chunk.text).join(Document, Chunk.document_id == Document.id)
            .where(Document.source_id == source.id, Chunk.chunk_kind == "leaf").order_by(Document.id, Chunk.chunk_index).limit(1)
        ).scalar_one_or_none()
        return [{"id": 0, "title": source.name, "kind": "source", "concepts": list(concepts), "excerpt": (first or "")[:600]}]
    ids = [section.id for section in sections]
    concepts: dict[int, list[str]] = {}
    for section_id, name, _count in db.execute(
        select(Chunk.section_id, Concept.canonical_name, func.count(Evidence.id))
        .join(Evidence, Evidence.chunk_id == Chunk.id).join(Concept, Evidence.concept_id == Concept.id)
        .where(Chunk.section_id.in_(ids)).group_by(Chunk.section_id, Concept.id)
        .order_by(Chunk.section_id, func.count(Evidence.id).desc(), Concept.canonical_name)
    ).all():
        names = concepts.setdefault(section_id, [])
        if len(names) < 8:
            names.append(name)
    densest: dict[int, tuple[int, str]] = {}
    for section_id, chunk_text, links in db.execute(
        select(Chunk.section_id, Chunk.text, func.count(Evidence.id))
        .join(Evidence, Evidence.chunk_id == Chunk.id).where(Chunk.section_id.in_(ids)).group_by(Chunk.id)
    ).all():
        if section_id not in densest or links > densest[section_id][0]:
            densest[section_id] = (links, chunk_text)
    return [{"id": section.id, "title": section.title, "kind": section.kind, "concepts": concepts.get(section.id, []),
             "excerpt": " ".join(densest.get(section.id, (0, ""))[1].split())[:240]} for section in sections]


def _source_data(db: Session, sources: list[Source]) -> list[dict]:
    data = []
    for source in sources:
        sections = _section_rows(db, source)
        limit = max(1, settings.plan_section_limit)
        if len(sections) > limit:
            logging.info("Source %s has %s sections; the plan describes the first %s", source.id, len(sections), limit)
        data.append({
            "id": source.id, "name": source.name, "uri": source.uri, "user_space_id": source.course_id,
            "profile": _profile(source), "sections": sections[:limit], "omitted_sections": max(0, len(sections) - limit),
        })
    return data


FINGERPRINT_PROFILE_FIELDS = ("course_title", "primary_learning_goal", "domain", "estimated_level", "source_role", "learning_objectives")


def source_fingerprint(item: dict) -> str:
    """What a source's outline is built from: its key profile fields, sections and their concepts.

    Status changes, filing and other row updates do not change it, so plans are only rebuilt
    when the content they describe changes.
    """
    basis = {"profile": {key: item["profile"].get(key) for key in FINGERPRINT_PROFILE_FIELDS},
             "sections": [[s["id"], s["title"], s["concepts"]] for s in item["sections"]]}
    return _digest(basis)


def _candidate_pairs(db: Session, source_ids: list[int], k: int) -> set[tuple[int, int]] | None:
    """Pairs worth an AI comparison: each source with its k most similar sources (by mean embedding).

    Returns None (compare everything) when there are few sources or no embeddings.
    """
    if len(source_ids) <= k + 1:
        return None
    centroids: dict[int, list[float]] = {}
    for source_id in source_ids:
        vectors = []
        for (embedding_json,) in db.execute(
            select(Chunk.embedding_json).join(Document, Chunk.document_id == Document.id)
            .where(Document.source_id == source_id, Chunk.chunk_kind == "leaf", Chunk.embedding_json.is_not(None))
            .order_by(Chunk.id).limit(200)
        ).all():
            try:
                vectors.append(json.loads(embedding_json))
            except json.JSONDecodeError:
                continue
        if vectors and all(len(v) == len(vectors[0]) for v in vectors):
            centroids[source_id] = [sum(values) / len(vectors) for values in zip(*vectors)]
    if centroids:
        # A source embedded with a different model cannot be ranked; treat it as unembedded.
        dimensions: dict[int, int] = {}
        for centroid in centroids.values():
            dimensions[len(centroid)] = dimensions.get(len(centroid), 0) + 1
        common = max(dimensions, key=lambda size: dimensions[size])
        centroids = {sid: centroid for sid, centroid in centroids.items() if len(centroid) == common}
    if len(centroids) < 2:
        return None
    pairs: set[tuple[int, int]] = set()
    for source_id, centroid in centroids.items():
        scored = sorted(((cosine(centroid, other), other_id) for other_id, other in centroids.items()
                         if other_id != source_id and len(other) == len(centroid)), reverse=True)
        pairs.update(tuple(sorted((source_id, other_id))) for _, other_id in scored[:k])
    # Sources without embeddings cannot be ranked; compare them with everything.
    for source_id in set(source_ids) - set(centroids):
        pairs.update(tuple(sorted((source_id, other))) for other in source_ids if other != source_id)
    return pairs


def _classify_relationships(db: Session, ai: AIProvider, sources: list[dict], warnings: list[str] | None = None,
                            candidates: set[tuple[int, int]] | None = None) -> list[dict]:
    """Pairwise course identity. Only `candidates` (similar pairs) are sent to the AI when given;
    other pairs are simply not merged, which is what an AI comparison of dissimilar sources returns."""
    warnings = [] if warnings is None else warnings
    by_id = {row["id"]: row for row in sources}
    ids = sorted(by_id)
    pairs = list(itertools.combinations(ids, 2))
    existing: dict[tuple[int, int], SourceRelationship] = {}
    # Outdated rows are rewritten in place rather than deleted and re-added: in one flush
    # SQLAlchemy inserts before it deletes, which would break the unique pair.
    stale: dict[tuple[int, int], SourceRelationship] = {}
    for row in db.execute(select(SourceRelationship).where(SourceRelationship.source_a_id.in_(ids), SourceRelationship.source_b_id.in_(ids))).scalars().all():
        expected_hashes = (_digest(by_id[row.source_a_id]["profile"]), _digest(by_id[row.source_b_id]["profile"]))
        if expected_hashes != (row.source_a_profile_hash, row.source_b_profile_hash) or _is_unclassified_default(row):
            stale[(row.source_a_id, row.source_b_id)] = row
        else:
            existing[(row.source_a_id, row.source_b_id)] = row
    missing = [pair for pair in pairs if pair not in existing and (candidates is None or pair in candidates)]
    provisional: list[dict] = []
    if missing:
        identity_fields = ("author", "institution", "provider", "source_type", "domain", "subdomain", "course_title",
                           "course_identity_confidence", "primary_learning_goal", "learning_intent_confidence",
                           "learning_objectives", "estimated_level", "source_role", "web_evidence")
        profile_summaries = [{"source_id": sid, "title": by_id[sid]["name"], "uri": by_id[sid]["uri"],
                              "user_space_id": by_id[sid]["user_space_id"],
                              "profile": {key: by_id[sid]["profile"].get(key) for key in identity_fields}}
                             for sid in ids]
        prompt = f"""Classify these source pairs by course and learning-path identity before merging curricula.
Topic overlap alone is never enough. Consider course title/code, author, institution, source role, target level, stated goal/outcomes, and trustworthy source evidence. The user's selected learning space is only a weak clue, not proof. If evidence is uncertain, classify RELATED_BUT_SEPARATE or UNRELATED and lower confidence.
Allowed types: SAME_COURSE, SAME_LEARNING_PATH, COMPLEMENTARY_RESOURCE, SAME_DOMAIN_DIFFERENT_COURSE, RELATED_BUT_SEPARATE, UNRELATED.
SAME_COURSE means the sources identify one named course. SAME_LEARNING_PATH means compatible goals and learner outcomes. COMPLEMENTARY_RESOURCE means the resource supports the same objective without being the course itself. A deployment tutorial and an NLP or RAG course stay separate unless their stated objectives explicitly unite them.
Sources: {json.dumps(profile_summaries, ensure_ascii=False)}
Requested pairs: {json.dumps(missing)}
Return JSON relationships with source_a_id, source_b_id, type, confidence (0..1), rationale, and evidence (brief facts or URLs). Do not return prerequisite edges."""
        schema = {"type": "object", "properties": {"relationships": {"type": "array", "items": {"type": "object", "properties": {
            "source_a_id": {"type": "integer"}, "source_b_id": {"type": "integer"},
            "type": {"type": "string", "enum": sorted(RELATION_TYPES)}, "confidence": {"type": "number"},
            "rationale": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}},
        }, "required": ["source_a_id", "source_b_id", "type", "confidence", "rationale", "evidence"]}}}, "required": ["relationships"]}
        try:
            result = ai.generate_json(prompt, task="source_relationship", schema=schema)
            generated = {}
            for item in result.get("relationships", []):
                a, b = sorted((int(item["source_a_id"]), int(item["source_b_id"])))
                generated[(a, b)] = item
        except ProviderQuotaExceededError:
            raise
        except Exception:
            generated = {}
            warnings.append("The AI service was unavailable while comparing sources, so they are shown as separate paths for now.")
        for a, b in missing:
            item = generated.get((a, b))
            if item is None:
                # Nothing was learned about this pair. Show it as separate in this plan only:
                # saving the guess would stop the pair from ever being compared again.
                provisional.append({"source_a_id": a, "source_b_id": b, "relation_type": "RELATED_BUT_SEPARATE",
                                    "confidence": 0.2, "rationale": _UNCLASSIFIED_RATIONALE, "evidence": [], "provisional": True})
                continue
            relation = str(item.get("type", "RELATED_BUT_SEPARATE")).upper()
            if relation not in RELATION_TYPES:
                relation = "RELATED_BUT_SEPARATE"
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence", 0.2))))
            except (TypeError, ValueError):
                confidence = 0.2
            row = stale.pop((a, b), None)
            if row is None:
                row = SourceRelationship(source_a_id=a, source_b_id=b)
                db.add(row)
            row.source_a_profile_hash = _digest(by_id[a]["profile"])
            row.source_b_profile_hash = _digest(by_id[b]["profile"])
            row.relation_type = relation
            row.confidence = confidence
            row.rationale = str(item.get("rationale", "Not enough evidence to confirm a shared learning path."))[:2000]
            row.evidence_json = json.dumps(item.get("evidence", []), ensure_ascii=False)
            existing[(a, b)] = row
        if provisional and not any("comparing sources" in w for w in warnings):
            warnings.append(f"{len(provisional)} source pair(s) could not be compared yet and are shown as separate.")
        # Keep relationship writes pending until the caller persists the completed
        # plan. Flushing here would hold SQLite's single-writer lock through the
        # independent curriculum and synthesis model calls below.
    for row in stale.values():  # outdated and not re-answered (or no longer worth comparing)
        db.delete(row)
    return provisional + [{"source_a_id": row.source_a_id, "source_b_id": row.source_b_id, "relation_type": row.relation_type,
             "confidence": row.confidence, "rationale": row.rationale, "evidence": json.loads(row.evidence_json or "[]")}
            for row in existing.values()]


def _node_schema() -> dict:
    node = {"type": "object", "properties": {
        "id": {"type": "string"}, "parent_id": {"type": "string"}, "type": {"type": "string", "enum": ["outline", "study"]},
        "title": {"type": "string"}, "purpose": {"type": "string"}, "prerequisites": {"type": "string"},
        "prerequisite_confidence": {"type": "number"}, "why_included": {"type": "string"}, "enables": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "integer"}}, "source_roles": {"type": "array", "items": {"type": "string"}},
        "topic_merge_confidence": {"type": "number"}, "core_or_supplementary": {"type": "string"},
        "comes_after": {"type": "array", "items": {"type": "string"}},
        "section_ids": {"type": "array", "items": {"type": "integer"}},
    }, "required": ["id", "parent_id", "type", "title", "purpose", "prerequisites", "prerequisite_confidence", "why_included", "enables", "source_ids", "source_roles", "topic_merge_confidence", "core_or_supplementary", "comes_after", "section_ids"]}
    return node


def _curriculum_schema() -> dict:
    node = _node_schema()
    return {"type": "object", "properties": {"nodes": {"type": "array", "items": node}, **{
        key: {"type": "array", "items": {"type": "string"}} for key in ("knowledge", "skills", "capabilities", "applications")
    }}, "required": ["nodes", "knowledge", "skills", "capabilities", "applications"]}


def _namespaced(nodes: list, source_id: int) -> list[dict]:
    """Per-source node ids are local ("1", "2" in every source): prefix them before mixing sources."""
    out = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        copy = dict(node)
        copy["id"] = f"s{source_id}:{node.get('id')}"
        parent = str(node.get("parent_id") or "none")
        copy["parent_id"] = parent if parent.casefold() in {"", "none", "root"} else f"s{source_id}:{parent}"
        out.append(copy)
    return out


def _normalize_nodes(raw_nodes: list[dict], path_id: str, source_ids: set[int], roles: dict[int, str],
                     previous_ids: frozenset[str] = frozenset(), section_ids: frozenset[int] = frozenset(),
                     used: set[str] | None = None, previous_by_sections: dict | None = None,
                     section_titles: dict[int, str] | None = None) -> list[dict]:
    """Validate model output and give every topic a stable id, in three passes over the outline:

    1. an id the model reused from the previous outline;
    2. the id of the previous topic of the same kind over exactly the same sections (matched by
       section id and title, since ids are recycled after a source is removed), preferring the
       topic with the same title when several claim it;
    3. an id from the title, with a counter if it is taken; a topic is never dropped.
    `used` is shared by all paths of a plan, so no id can appear twice.
    """
    used = set() if used is None else used
    used.add(path_id)  # the path root's id is taken
    by_sections = previous_by_sections or {}
    titles_of = section_titles or {}
    entries = []
    seen_titles: set[str] = set()
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title", "")).strip()
        local_id = str(raw.get("id", "")).strip()
        kind = raw.get("type")
        if not title or not local_id or kind not in {"outline", "study"} or title.casefold() in seen_titles:
            continue
        seen_titles.add(title.casefold())
        raw_sections = raw.get("section_ids") if isinstance(raw.get("section_ids"), list) else []
        sections = list(dict.fromkeys(x for x in raw_sections if isinstance(x, int) and not isinstance(x, bool) and x in section_ids))
        entries.append({"raw": raw, "title": title, "local_id": local_id, "kind": kind, "sections": sections, "id": None})

    for entry in entries:
        local_id = entry["local_id"]
        if local_id in previous_ids and local_id not in used and not local_id.startswith("path-"):
            entry["id"] = local_id
            used.add(local_id)
    claims: dict[str, list[tuple[dict, str]]] = {}
    for entry in entries:
        if entry["id"] is None and entry["sections"]:
            match = by_sections.get((entry["kind"], frozenset((sid, titles_of.get(sid, "")) for sid in entry["sections"])))
            if match and match[0] not in used:
                claims.setdefault(match[0], []).append((entry, match[1]))
    for previous_id, claimants in claims.items():
        same_title = [entry for entry, previous_title in claimants if entry["title"].casefold() == previous_title.casefold()]
        chosen = same_title[0] if same_title else (claimants[0][0] if len(claimants) == 1 else None)
        if chosen is not None:
            chosen["id"] = previous_id
            used.add(previous_id)
    for entry in entries:
        if entry["id"] is None:
            stem = f"{path_id}:{entry['title'].casefold()}"
            node_id = "t-" + hashlib.sha1(stem.encode()).hexdigest()[:12]
            attempt = 0
            while node_id in used:  # a reused id took this title's id: derive another, never drop the topic
                attempt += 1
                node_id = "t-" + hashlib.sha1(f"{stem}:{attempt}".encode()).hexdigest()[:12]
            entry["id"] = node_id
            used.add(node_id)

    def confidence(value) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    id_map: dict[str, str] = {}
    nodes = []
    for entry in entries:
        raw = entry["raw"]
        id_map[entry["local_id"]] = entry["id"]
        cited_ids = [int(x) for x in raw.get("source_ids", []) if str(x).isdigit() and int(x) in source_ids]
        nodes.append({"node_id": entry["id"], "local_parent": str(raw.get("parent_id") or "none"), "node_type": entry["kind"],
                      "title": entry["title"], "source_ids": cited_ids,
                      "source_roles": list(dict.fromkeys(str(x) for x in raw.get("source_roles", []))) or [roles[sid] for sid in cited_ids],
                      "purpose": str(raw.get("purpose", "")), "prerequisites": str(raw.get("prerequisites", "")),
                      "prerequisite_confidence": confidence(raw.get("prerequisite_confidence", 0.0)),
                      "why_included": str(raw.get("why_included", "")), "enables": str(raw.get("enables", "")),
                      "topic_merge_confidence": confidence(raw.get("topic_merge_confidence", 0.0)),
                      "core_or_supplementary": str(raw.get("core_or_supplementary", "core")),
                      "comes_after": [str(x) for x in raw.get("comes_after", [])],
                      "section_ids": entry["sections"],
                      "section_titles": [titles_of.get(sid, "") for sid in entry["sections"]]})
    for node in nodes:
        node["parent_node_id"] = id_map.get(node.pop("local_parent"))
    return nodes


class PlanQuotaExhausted(RuntimeError):
    """The daily AI quota ran out while rebuilding an existing plan; the previous plan is kept."""


def _has_study_topic(curriculum) -> bool:
    return isinstance(curriculum, dict) and any(
        isinstance(node, dict) and node.get("type") == "study" and str(node.get("title") or "").strip() and str(node.get("id") or "").strip()
        for node in curriculum.get("nodes") or [])


def _source_curriculum(db: Session, ai: AIProvider, item: dict, warnings: list[str], heartbeat=None, *,
                       allow_ai: bool = True) -> tuple[dict, bool]:
    """This source's own outline: the saved one while its fingerprint is unchanged, else a new AI
    call (unless `allow_ai` is False, e.g. after the daily quota ran out). Returns (curriculum, reused)."""
    source = db.get(Source, item["id"])
    fingerprint = source_fingerprint(item)
    if source is not None and source.curriculum_hash == fingerprint and source.curriculum_json:
        try:
            saved = json.loads(source.curriculum_json)
            if _has_study_topic(saved):
                return saved, True
        except json.JSONDecodeError:
            pass
    if not allow_ai:
        return {"nodes": [], "knowledge": [], "skills": [], "capabilities": [], "applications": []}, False
    sid = item["id"]
    sections = [{"id": s["id"], "title": s["title"], "kind": s["kind"], "concepts": s["concepts"], "excerpt": s["excerpt"]} for s in item["sections"]]
    omitted = f" ({item['omitted_sections']} later sections are not listed.)" if item.get("omitted_sections") else ""
    prompt = f"""Extract the learning curriculum for this single source. Do not merge it with any other source. Only include topics supported by its sections and concepts.
Group the sections into teachable topics in a sensible learning order. A topic may cover one section or several consecutive ones; split a broad section only when its concepts clearly form separate topics. Mark a node "outline" when it has child topics and "study" when it is a teachable unit. Keep the hierarchy as shallow as the material allows.
A prerequisite must be needed for this source's stated objective and supported by its content; semantic similarity or general usefulness is not enough.
Source profile: {json.dumps(item['profile'], ensure_ascii=False)}
Sections in reading order (id, title, kind, key concepts, short excerpt){omitted}: {json.dumps(sections, ensure_ascii=False)}
Return nodes plus knowledge, skills, capabilities, and applications. Every node must include section_ids (the ids of the sections it covers), why_included, enables, source_ids [{sid}], source_roles, prerequisite_confidence, topic_merge_confidence, and core_or_supplementary. Keep uncertain claims low-confidence. Node IDs are local to this source; parent_id is none for roots."""
    if heartbeat is not None:
        heartbeat()
    try:
        curriculum = ai.generate_json(prompt, task="syllabus", schema=_curriculum_schema())
    except ProviderQuotaExceededError:
        raise
    except Exception:
        warnings.append(f"Could not extract topics from '{item['name']}' because the AI service returned an error.")
        return {"nodes": [], "knowledge": [], "skills": [], "capabilities": [], "applications": []}, False
    if not isinstance(curriculum, dict) or not isinstance(curriculum.get("nodes"), list):
        curriculum = {"nodes": [], "knowledge": [], "skills": [], "capabilities": [], "applications": []}
    if source is not None and _has_study_topic(curriculum):  # an outline without study topics is retried next time
        source.curriculum_json = json.dumps(curriculum, ensure_ascii=False)
        source.curriculum_hash = fingerprint
    return curriculum, False


def build_syllabus_structured(db: Session, ai: AIProvider, course_id: int | None = None, *,
                              previous_nodes: list[dict] | None = None, heartbeat=None) -> tuple[str, list[dict]]:
    """Build a study plan for a scope. Each source's outline is reused while its content is
    unchanged, so adding one source costs that source's outline plus one merge call."""
    query = select(Source).where(Source.status == "done").order_by(Source.id)
    if course_id is not None:
        query = query.where(Source.course_id == course_id)
    source_rows = db.execute(query).scalars().all()
    if not source_rows:
        return "No completed sources are available in this scope yet.", []
    sources = _source_data(db, source_rows)
    warnings: list[str] = []
    if heartbeat is not None:
        heartbeat()
    candidates = _candidate_pairs(db, [item["id"] for item in sources], max(1, settings.plan_compare_k))
    quota_out = False
    quota_warning = "The AI service's daily quota is used up, so some sources are not in this plan yet. Press Refresh after the quota resets."
    try:
        relationships = _classify_relationships(db, ai, sources, warnings, candidates)
    except ProviderQuotaExceededError:
        if previous_nodes:
            raise PlanQuotaExhausted("The AI service's daily quota is used up, so the previous plan is kept. Press Refresh after the quota resets.")
        quota_out = True
        warnings.append(quota_warning)
        relationships = []  # not compared yet: shown as separate paths in this first plan
    db.commit()  # save comparisons now: a later failure must not throw away calls already paid for
    source_paths = group_source_paths([item["id"] for item in sources], relationships)
    by_id = {item["id"]: item for item in sources}

    # Extract each source curriculum independently before considering any merge.
    independent = []
    reused = 0
    for item in sources:
        sid, profile = item["id"], item["profile"]
        curriculum = {"nodes": [], "knowledge": [], "skills": [], "capabilities": [], "applications": []}
        if item.get("omitted_sections"):
            total = len(item["sections"]) + item["omitted_sections"]
            warnings.append(f"Only the first {len(item['sections'])} of {total} sections of '{item['name']}' are in this plan.")
        try:
            # Saved outlines are still used after the quota runs out; only new AI calls stop.
            curriculum, was_reused = _source_curriculum(db, ai, item, warnings, heartbeat, allow_ai=not quota_out)
            reused += was_reused
            if not was_reused and not quota_out:
                db.commit()  # keep this source's new outline even if a later step fails
        except ProviderQuotaExceededError:
            quota_out = True
            warnings.append(quota_warning)
        independent.append({"source_id": sid, "title": item["name"], "profile": profile, **curriculum})
    if quota_out and previous_nodes:
        # Replacing a complete plan with a partial one would hide topics and progress.
        raise PlanQuotaExhausted("The AI service's daily quota is used up, so the previous plan is kept. Press Refresh after the quota resets.")
    db.info["syllabus_reused_sources"] = reused

    path_inputs = []
    for index, member_ids in enumerate(source_paths, 1):
        members = [{**row, "nodes": _namespaced(row.get("nodes", []), row["source_id"])} for row in independent if row["source_id"] in member_ids]
        titles = [row["profile"].get("course_title") for row in members if row["profile"].get("course_title")]
        goals = [row["profile"].get("primary_learning_goal") for row in members if row["profile"].get("primary_learning_goal")]
        relationships_in_path = [rel for rel in relationships if rel["source_a_id"] in member_ids and rel["source_b_id"] in member_ids]
        path_inputs.append({"path_id": f"P{index}", "source_ids": member_ids, "suggested_title": titles[0] if titles else (goals[0] if goals else members[0]["title"]), "source_curricula": members, "relationships": relationships_in_path})

    schema = {"type": "object", "properties": {
        "paths": {"type": "array", "items": {"type": "object", "properties": {
            "path_id": {"type": "string"}, "title": {"type": "string"}, "source_ids": {"type": "array", "items": {"type": "integer"}},
            "primary_goal": {"type": "string"}, **{key: {"type": "array", "items": {"type": "string"}} for key in ("knowledge", "skills", "capabilities", "applications")},
            "nodes": {"type": "array", "items": _node_schema()},
        }, "required": ["path_id", "title", "source_ids", "primary_goal", "knowledge", "skills", "capabilities", "applications", "nodes"]}},
        "cross_path_relations": {"type": "array", "items": {"type": "object", "properties": {"from_path": {"type": "string"}, "to_path": {"type": "string"}, "type": {"type": "string"}, "rationale": {"type": "string"}, "confidence": {"type": "number"}}, "required": ["from_path", "to_path", "type", "rationale", "confidence"]}},
        "shared_foundations": {"type": "array", "items": {"type": "string"}},
    }, "required": ["paths", "cross_path_relations", "shared_foundations"]}
    previous = [{"id": n.get("id"), "title": n.get("title"), "parent_id": n.get("parent_id"), "type": n.get("type")}
                for n in (previous_nodes or []) if not str(n.get("id") or "").startswith("path-")][:300]
    keep_ids = f"""
Existing outline from the last update. When a topic still applies, return it with the same id so the learner keeps their progress; drop topics whose sources are gone: {json.dumps(previous, ensure_ascii=False)}""" if previous else ""
    prompt = f"""Synthesize these independent source curricula into one or more learning paths.
Never merge topics across different path_id values. Within one path, merge topics only if their learning purpose is equivalent; sharing a word (such as embeddings) is not enough. Preserve every source's objective and core coverage. Do not infer prerequisites from semantic similarity or general usefulness. A prerequisite must be required for this path's outcomes and source-supported; otherwise leave it empty or describe a relationship without asserting a dependency. Preserve low confidence and disagreements.
Return path metadata for knowledge, skills, capabilities, practical applications, and source coverage. Explain why every canonical topic belongs, what capability it enables, which source introduced it, whether it is core or supplementary, and what comes after it. Cross-path relations may be related_to, complementary_to, or shared_foundation only; they are never prerequisites.
Independent curricula and relationship evidence: {json.dumps(path_inputs, ensure_ascii=False)}
Source identity classifications: {json.dumps(relationships, ensure_ascii=False)}{keep_ids}"""
    single = len(path_inputs) == 1 and len(path_inputs[0]["source_ids"]) == 1
    try:
        if quota_out:
            raise RuntimeError("daily quota used up")
        if single:
            # One source in one path: its own outline is the plan; a merge call would add nothing.
            curriculum = path_inputs[0]["source_curricula"][0]
            result = {"paths": [{"path_id": path_inputs[0]["path_id"], "title": path_inputs[0]["suggested_title"],
                                 "source_ids": path_inputs[0]["source_ids"],
                                 "primary_goal": str(curriculum.get("profile", {}).get("primary_learning_goal") or path_inputs[0]["suggested_title"]),
                                 **{key: curriculum.get(key, []) for key in ("knowledge", "skills", "capabilities", "applications")},
                                 "nodes": curriculum.get("nodes", [])}],
                      "cross_path_relations": [], "shared_foundations": []}
        else:
            if heartbeat is not None:
                heartbeat()
            result = ai.generate_json(prompt, task="syllabus", schema=schema)
        merge_failed = False
    except ProviderQuotaExceededError:
        if previous_nodes:
            raise PlanQuotaExhausted("The AI service's daily quota is used up, so the previous plan is kept. Press Refresh after the quota resets.")
        merge_failed = True
    except Exception:
        merge_failed = True
    if merge_failed:
        warnings.append("Could not merge the sources into one plan, so each source's own topics are shown.")
        result = {"paths": [{"path_id": path["path_id"], "title": path["suggested_title"], "source_ids": path["source_ids"],
                  "primary_goal": path["suggested_title"], "knowledge": [], "skills": [], "capabilities": [], "applications": [],
                  "nodes": [node for item in path["source_curricula"] for node in item.get("nodes", [])]} for path in path_inputs],
                  "cross_path_relations": [], "shared_foundations": []}

    # Treat model output as untrusted: it may only return the exact groups
    # established by the persisted pairwise identity classifications.
    expected_paths = {path["path_id"]: path for path in path_inputs}
    returned_paths = {}
    for candidate in result.get("paths", []):
        key = str(candidate.get("path_id", ""))
        expected = expected_paths.get(key)
        if expected is None or key in returned_paths:
            continue
        allowed = set(expected["source_ids"])
        candidate["source_ids"] = [int(sid) for sid in candidate.get("source_ids", [])
                                   if str(sid).isdigit() and int(sid) in allowed]
        # Identity classification, not generated output, owns path membership.
        candidate["source_ids"] = expected["source_ids"]
        returned_paths[key] = candidate
    for path in path_inputs:
        if path["path_id"] not in returned_paths:
            returned_paths[path["path_id"]] = {
                "path_id": path["path_id"], "title": path["suggested_title"],
                "source_ids": path["source_ids"], "primary_goal": path["suggested_title"],
                "knowledge": [], "skills": [], "capabilities": [], "applications": [],
                "nodes": [node for source_curriculum in path["source_curricula"]
                          for node in source_curriculum.get("nodes", [])],
            }

    source_names = {item["id"]: item["name"] for item in sources}
    source_roles = {item["id"]: str(item["profile"].get("source_role") or "unknown") for item in sources}
    persisted = []
    member_sets = [{int(i) for i in path.get("source_ids", []) if str(i).isdigit() and int(i) in source_names} for path in returned_paths.values()]
    used_ids: set[str] = {f"path-{min(members)}" for members in member_sets if members}
    section_titles = {section["id"]: section["title"] for item in sources for section in item["sections"]}
    section_keys: dict = {}
    for node in previous_nodes or []:
        secs = node.get("section_ids") or []
        names = node.get("section_titles") or []
        if secs and len(names) == len(secs) and not str(node.get("id") or "").startswith("path-"):
            key = (node.get("type"), frozenset(zip(secs, names)))
            # Ambiguous matches are not used.
            section_keys[key] = None if key in section_keys else (node.get("id"), str(node.get("title") or ""))
    previous_by_sections = {key: value for key, value in section_keys.items() if value}
    for path in returned_paths.values():
        member_ids = {int(item) for item in path.get("source_ids", []) if str(item).isdigit() and int(item) in source_names}
        if not member_ids:
            continue
        path_key = f"path-{min(member_ids)}"
        title = str(path.get("title") or expected_paths[path["path_id"]]["suggested_title"])[:500]
        ids = sorted(member_ids)
        related = [rel for rel in relationships if rel["source_a_id"] in member_ids and rel["source_b_id"] in member_ids]
        root_meta = {"purpose": str(path.get("primary_goal", "")), "prerequisites": "", "parallel": "",
                     "source_groups": [source_names[sid] for sid in ids], "source_ids": ids,
                     "source_roles": [source_roles[sid] for sid in ids], "why_included": "Sources are grouped because their course identity or learning objectives are compatible.",
                     "enables": "; ".join(path.get("capabilities", [])), "core_or_supplementary": "learning path",
                     "knowledge": path.get("knowledge", []), "skills": path.get("skills", []),
                     "capabilities": path.get("capabilities", []), "applications": path.get("applications", []),
                     "source_relationships": related}
        persisted.append({"node_id": path_key, "parent_node_id": None, "node_type": "outline", "path_id": path_key, "title": title, "metadata_json": json.dumps(root_meta, ensure_ascii=False)})
        valid_sections = frozenset(section["id"] for sid in member_ids for section in by_id[sid]["sections"])
        nodes = _normalize_nodes(path.get("nodes", []), path_key, member_ids, source_roles,
                                 frozenset(n["id"] for n in previous if n.get("id")), valid_sections, used_ids, previous_by_sections,
                                 section_titles)
        node_ids = {node["node_id"] for node in nodes}
        for node in nodes:
            parent_id = node.get("parent_node_id") if node.get("parent_node_id") in node_ids else path_key
            if parent_id == node["node_id"]:
                parent_id = path_key
            metadata = {key: value for key, value in node.items() if key not in {"node_id", "parent_node_id", "node_type", "title"}}
            metadata["source_groups"] = [source_names[sid] for sid in node["source_ids"] if sid in source_names]
            persisted.append({"node_id": node["node_id"], "parent_node_id": parent_id, "node_type": node["node_type"], "path_id": path_key, "title": node["title"], "metadata_json": json.dumps(metadata, ensure_ascii=False)})
    if not persisted:
        return "No valid learning paths could be generated from these source profiles.", []

    rows = ["| # | ID | Parent ID | Topic | Type | Purpose | Prerequisites | Parallel approaches | Source groups |", "|---|---|---|---|---|---|---|---|---|---|"]
    for number, node in enumerate(persisted, 1):
        meta = json.loads(node["metadata_json"] or "{}")
        values = [str(number), node["node_id"], node["parent_node_id"] or "none", node["title"], node["node_type"], meta.get("purpose", ""), meta.get("prerequisites", ""), meta.get("parallel", ""), ", ".join(meta.get("source_groups", []))]
        rows.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |")
    if not any(node["node_type"] == "study" for node in persisted):
        warnings.append("No study topics could be extracted from these sources yet.")
    db.info["syllabus_warnings"] = list(dict.fromkeys(warnings))
    db.info["syllabus_cross_path_relations"] = result.get("cross_path_relations", [])
    db.info["syllabus_shared_foundations"] = result.get("shared_foundations", [])
    db.info["syllabus_source_relationships"] = relationships
    return "\n".join(rows), persisted
