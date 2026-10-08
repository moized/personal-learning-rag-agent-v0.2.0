"""Source-first course identity and curriculum synthesis."""
import hashlib
import itertools
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Chunk, Concept, Document, Evidence, Source, SourceRelationship
from ..providers.base import AIProvider

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


def _source_data(db: Session, sources: list[Source]) -> list[dict]:
    data = []
    for source in sources:
        chunks = db.execute(
            select(Chunk).join(Document, Chunk.document_id == Document.id)
            .where(Document.source_id == source.id, Chunk.chunk_kind == "leaf", Chunk.embedding_status == "done")
            .order_by(Chunk.chunk_index).limit(8)
        ).scalars().all()
        concepts = db.execute(
            select(Concept.canonical_name).join(Evidence, Evidence.concept_id == Concept.id)
            .join(Chunk, Evidence.chunk_id == Chunk.id).join(Document, Chunk.document_id == Document.id)
            .where(Document.source_id == source.id, Chunk.chunk_kind == "leaf").distinct().limit(60)
        ).all()
        data.append({
            "id": source.id, "name": source.name, "uri": source.uri, "user_space_id": source.course_id,
            "profile": _profile(source), "concepts": [name for (name,) in concepts],
            "evidence": [{"chunk_id": row.id, "text": row.text[:1300]} for row in chunks],
        })
    return data


def _classify_relationships(db: Session, ai: AIProvider, sources: list[dict], warnings: list[str] | None = None) -> list[dict]:
    warnings = [] if warnings is None else warnings
    by_id = {row["id"]: row for row in sources}
    ids = sorted(by_id)
    pairs = list(itertools.combinations(ids, 2))
    existing: dict[tuple[int, int], SourceRelationship] = {}
    stale = []
    for row in db.execute(select(SourceRelationship).where(SourceRelationship.source_a_id.in_(ids), SourceRelationship.source_b_id.in_(ids))).scalars().all():
        expected_hashes = (_digest(by_id[row.source_a_id]["profile"]), _digest(by_id[row.source_b_id]["profile"]))
        if expected_hashes != (row.source_a_profile_hash, row.source_b_profile_hash) or _is_unclassified_default(row):
            stale.append(row)
        else:
            existing[(row.source_a_id, row.source_b_id)] = row
    for row in stale:
        db.delete(row)
    missing = [pair for pair in pairs if pair not in existing]
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
            row = SourceRelationship(
                source_a_id=a, source_b_id=b, source_a_profile_hash=_digest(by_id[a]["profile"]),
                source_b_profile_hash=_digest(by_id[b]["profile"]), relation_type=relation,
                confidence=confidence, rationale=str(item.get("rationale", "Not enough evidence to confirm a shared learning path."))[:2000],
                evidence_json=json.dumps(item.get("evidence", []), ensure_ascii=False),
            )
            db.add(row)
            existing[(a, b)] = row
        if provisional and not any("comparing sources" in w for w in warnings):
            warnings.append(f"{len(provisional)} source pair(s) could not be compared yet and are shown as separate.")
        # Keep relationship writes pending until the caller persists the completed
        # plan. Flushing here would hold SQLite's single-writer lock through the
        # independent curriculum and synthesis model calls below.
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
    }, "required": ["id", "parent_id", "type", "title", "purpose", "prerequisites", "prerequisite_confidence", "why_included", "enables", "source_ids", "source_roles", "topic_merge_confidence", "core_or_supplementary", "comes_after"]}
    return node


def _curriculum_schema() -> dict:
    node = _node_schema()
    return {"type": "object", "properties": {"nodes": {"type": "array", "items": node}, **{
        key: {"type": "array", "items": {"type": "string"}} for key in ("knowledge", "skills", "capabilities", "applications")
    }}, "required": ["nodes", "knowledge", "skills", "capabilities", "applications"]}


def _normalize_nodes(raw_nodes: list[dict], path_id: str, source_ids: set[int], roles: dict[int, str]) -> list[dict]:
    id_map: dict[str, str] = {}
    nodes = []
    seen_titles: set[str] = set()
    for raw in raw_nodes:
        title = str(raw.get("title", "")).strip()
        local_id = str(raw.get("id", "")).strip()
        kind = raw.get("type")
        if not title or not local_id or kind not in {"outline", "study"} or title.casefold() in seen_titles:
            continue
        seen_titles.add(title.casefold())
        node_id = "t-" + hashlib.sha1(f"{path_id}:{title.casefold()}".encode()).hexdigest()[:12]
        id_map[local_id] = node_id
        cited_ids = [int(x) for x in raw.get("source_ids", []) if str(x).isdigit() and int(x) in source_ids]
        def confidence(value) -> float:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                return 0.0
        nodes.append({"node_id": node_id, "local_parent": str(raw.get("parent_id") or "none"), "node_type": kind,
                      "title": title, "source_ids": cited_ids,
                      "source_roles": list(dict.fromkeys(str(x) for x in raw.get("source_roles", []))) or [roles[sid] for sid in cited_ids],
                      "purpose": str(raw.get("purpose", "")), "prerequisites": str(raw.get("prerequisites", "")),
                      "prerequisite_confidence": confidence(raw.get("prerequisite_confidence", 0.0)),
                      "why_included": str(raw.get("why_included", "")), "enables": str(raw.get("enables", "")),
                      "topic_merge_confidence": confidence(raw.get("topic_merge_confidence", 0.0)),
                      "core_or_supplementary": str(raw.get("core_or_supplementary", "core")),
                      "comes_after": [str(x) for x in raw.get("comes_after", [])]})
    for node in nodes:
        node["parent_node_id"] = id_map.get(node.pop("local_parent"))
    return nodes


def build_syllabus_structured(db: Session, ai: AIProvider, course_id: int | None = None) -> tuple[str, list[dict]]:
    query = select(Source).where(Source.status == "done").order_by(Source.id)
    if course_id is not None:
        query = query.where(Source.course_id == course_id)
    source_rows = db.execute(query).scalars().all()
    if not source_rows:
        return "No completed sources are available in this scope yet.", []
    sources = _source_data(db, source_rows)
    warnings: list[str] = []
    relationships = _classify_relationships(db, ai, sources, warnings)
    source_paths = group_source_paths([item["id"] for item in sources], relationships)
    by_id = {item["id"]: item for item in sources}

    # Extract each source curriculum independently before considering any merge.
    independent = []
    for item in sources:
        sid, profile = item["id"], item["profile"]
        prompt = f"""Extract the learning curriculum for this single source. Do not merge it with any other source. Only include topics supported by this source's concepts or evidence. A prerequisite must be needed for this source's stated objective and supported by its evidence; semantic similarity or general usefulness is not enough.
Source profile: {json.dumps(profile, ensure_ascii=False)}
Indexed concepts: {json.dumps(item['concepts'], ensure_ascii=False)}
Evidence excerpts: {json.dumps(item['evidence'], ensure_ascii=False)}
Return nodes plus knowledge, skills, capabilities, and applications. Every node must include why_included, enables, source_ids [{sid}], source_roles, prerequisite_confidence, topic_merge_confidence, and core_or_supplementary. Keep uncertain claims low-confidence. IDs are local to this source; parent_id is none for roots."""
        try:
            curriculum = ai.generate_json(prompt, task="syllabus", schema=_curriculum_schema())
        except Exception:
            warnings.append(f"Could not extract topics from '{item['name']}' because the AI service returned an error.")
            curriculum = {"nodes": [], "knowledge": [], "skills": [], "capabilities": [], "applications": []}
        independent.append({"source_id": sid, "title": item["name"], "profile": profile, **curriculum})

    path_inputs = []
    for index, member_ids in enumerate(source_paths, 1):
        members = [row for row in independent if row["source_id"] in member_ids]
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
    prompt = f"""Synthesize these independent source curricula into one or more learning paths.
Never merge topics across different path_id values. Within one path, merge topics only if their learning purpose is equivalent; sharing a word (such as embeddings) is not enough. Preserve every source's objective and core coverage. Do not infer prerequisites from semantic similarity or general usefulness. A prerequisite must be required for this path's outcomes and source-supported; otherwise leave it empty or describe a relationship without asserting a dependency. Preserve low confidence and disagreements.
Return path metadata for knowledge, skills, capabilities, practical applications, and source coverage. Explain why every canonical topic belongs, what capability it enables, which source introduced it, whether it is core or supplementary, and what comes after it. Cross-path relations may be related_to, complementary_to, or shared_foundation only; they are never prerequisites.
Independent curricula and relationship evidence: {json.dumps(path_inputs, ensure_ascii=False)}
Source identity classifications: {json.dumps(relationships, ensure_ascii=False)}"""
    try:
        result = ai.generate_json(prompt, task="syllabus", schema=schema)
    except Exception:
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
        nodes = _normalize_nodes(path.get("nodes", []), path_key, member_ids, source_roles)
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
