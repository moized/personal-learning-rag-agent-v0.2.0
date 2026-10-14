"""Organize chunks into semantic groups and tag the concepts each chunk teaches.

One model decision covers a whole batch of excerpts: it assigns each excerpt to an existing
group or a new, content-named group, and lists the concepts the excerpt teaches, reusing
existing concept names. In manual mode that is one paste per batch instead of one per chunk.
"""
import json
import re
from collections import defaultdict
import numpy as np
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from ..models import Chunk, Document, Group, Concept, Evidence, Source
from ..providers.base import AIProvider
from ..config import settings

LEGACY_GROUP_DESCRIPTION = "Automatically created semantic group"


def cosine(a: list[float], b: list[float]) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return 0.0 if den == 0 else float(np.dot(x, y) / den)


def normalize_name(name: str | None) -> str:
    """Case/punctuation/plural-insensitive key so 'Dense Retrievers' and 'dense retriever' match."""
    words = re.sub(r"[^\w]+", " ", (name or "").casefold()).split()
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words)


def _centroid(group: Group) -> list[float] | None:
    try:
        return json.loads(group.centroid_json) if group.centroid_json else None
    except (TypeError, json.JSONDecodeError):
        return None


def nearest_groups(db: Session, embedding: list[float], limit: int, threshold: float) -> list[tuple[Group, float]]:
    scored: list[tuple[Group, float]] = []
    for group in db.execute(select(Group)).scalars().all():
        centroid = _centroid(group)
        if not centroid or len(centroid) != len(embedding):
            continue
        score = cosine(embedding, centroid)
        if score >= threshold:
            scored.append((group, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


def candidate_groups(db: Session, embeddings: list[list[float]], limit: int) -> list[Group]:
    """Existing groups shown to the model: all of them, or the ones closest to this batch."""
    groups = db.execute(select(Group).order_by(Group.id)).scalars().all()
    if len(groups) <= limit or not embeddings:
        return groups
    matrix = np.asarray(embeddings, dtype=float)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    scored = []
    for group in groups:
        centroid = _centroid(group)
        if not centroid or len(centroid) != matrix.shape[1]:
            continue
        c = np.asarray(centroid, dtype=float)
        scored.append((float(np.max(matrix @ (c / max(np.linalg.norm(c), 1e-12)))), group))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [group for _, group in scored[:limit]]


class ConceptIndex:
    """Looks concepts up by normalized canonical name or alias, so synonyms resolve to one row."""

    def __init__(self, db: Session):
        self.db = db
        self.by_key: dict[str, Concept] = {}
        for concept in db.execute(select(Concept).order_by(Concept.id)).scalars().all():
            self._index(concept)

    def _index(self, concept: Concept) -> None:
        for name in [concept.canonical_name, *json.loads(concept.aliases_json or "[]")]:
            key = normalize_name(name)
            if key:
                self.by_key.setdefault(key, concept)

    def find(self, name: str | None) -> Concept | None:
        return self.by_key.get(normalize_name(name))

    def resolve(self, name: str, meta: dict | None, group: Group | None) -> Concept | None:
        name = str(name or "").strip()[:255]
        if not normalize_name(name):
            return None
        meta = meta if isinstance(meta, dict) else {}
        aliases = [str(a).strip() for a in meta.get("aliases") or [] if str(a).strip()] if isinstance(meta.get("aliases"), list) else []
        concept = self.find(name) or next((self.find(a) for a in aliases if self.find(a)), None)
        if concept is None:
            concept = Concept(canonical_name=name, description=str(meta.get("description") or "")[:2000] or None,
                              aliases_json=json.dumps(_dedupe_names(aliases, exclude=name), ensure_ascii=False),
                              group_id=group.id if group else None)
            self.db.add(concept)
            self.db.flush()
        else:
            if concept.group_id is None and group is not None:
                concept.group_id = group.id
            if not concept.description and meta.get("description"):
                concept.description = str(meta["description"])[:2000]
            old = json.loads(concept.aliases_json or "[]")
            concept.aliases_json = json.dumps(_dedupe_names([*old, name, *aliases], exclude=concept.canonical_name), ensure_ascii=False)
        self._index(concept)
        return concept


def _dedupe_names(names: list[str], exclude: str | None = None) -> list[str]:
    seen = {normalize_name(exclude)} if exclude else set()
    out = []
    for name in names:
        key = normalize_name(name)
        if key and key not in seen:
            seen.add(key)
            out.append(name)
    return out[:30]


def merge_concepts(db: Session, keep_name: str, merge_names: list[str]) -> int:
    """Fold concepts that name the same idea into one, moving their evidence and names into aliases."""
    index = ConceptIndex(db)
    keep = index.find(keep_name)
    if keep is None:
        return 0
    removed: set[int] = set()
    for name in merge_names:
        other = index.find(name)
        if other is None or other.id == keep.id or other.id in removed:
            continue
        for ev in db.execute(select(Evidence).where(Evidence.concept_id == other.id)).scalars().all():
            clash = db.execute(select(Evidence).where(Evidence.concept_id == keep.id, Evidence.chunk_id == ev.chunk_id)).scalar_one_or_none()
            if clash:
                db.delete(ev)
            else:
                ev.concept_id = keep.id
        names = [*json.loads(keep.aliases_json or "[]"), other.canonical_name, *json.loads(other.aliases_json or "[]")]
        keep.aliases_json = json.dumps(_dedupe_names(names, exclude=keep.canonical_name), ensure_ascii=False)
        keep.description = keep.description or other.description
        keep.group_id = keep.group_id or other.group_id
        db.flush()
        db.delete(other)
        db.flush()
        removed.add(other.id)
    return len(removed)


def organize_schema(with_concepts: bool) -> dict:
    assignment = {
        "excerpt": {"type": "integer"},
        "group": {"type": "string", "description": "exact name of an existing or new group"},
    }
    properties: dict = {
        "assignments": {
            "type": "array",
            "items": {"type": "object", "properties": assignment, "required": ["excerpt", "group"]},
        },
        "new_groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "short specific group name"},
                    "description": {"type": "string", "description": "one sentence"},
                    "keywords": {"type": "array", "items": {"type": "string", "description": "keyword"}},
                },
                "required": ["name", "description"],
            },
        },
    }
    if with_concepts:
        assignment["concepts"] = {"type": "array", "items": {"type": "string", "description": "concept name"}}
        properties["new_concepts"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "canonical concept name"},
                    "description": {"type": "string", "description": "one sentence"},
                    "aliases": {"type": "array", "items": {"type": "string", "description": "synonym or abbreviation"}},
                },
                "required": ["name", "description"],
            },
        }
    return {"type": "object", "properties": properties, "required": ["assignments"]}


def _excerpt_label(db: Session, chunk: Chunk) -> str:
    doc = db.get(Document, chunk.document_id)
    parts = [p for p in (doc.title if doc else None, doc.section if doc else None) if p]
    return " › ".join(parts)


def build_organize_prompt(db: Session, src: Source, hint: str | None, chunks: list[Chunk],
                          groups: list[Group], with_concepts: bool) -> str:
    concepts_by_group: dict[int, list[str]] = defaultdict(list)
    if with_concepts and groups:
        for concept in db.execute(
            select(Concept).where(Concept.group_id.in_([g.id for g in groups])).order_by(Concept.id)
        ).scalars().all():
            if len(concepts_by_group[concept.group_id]) < 25:
                concepts_by_group[concept.group_id].append(concept.canonical_name)
    group_lines = []
    for group in groups:
        line = f"- {group.name}"
        if group.description and group.description != LEGACY_GROUP_DESCRIPTION:
            line += f": {group.description}"
        if concepts_by_group.get(group.id):
            line += f" | concepts: {'; '.join(concepts_by_group[group.id])}"
        group_lines.append(line)

    excerpts = []
    for n, chunk in enumerate(chunks, 1):
        text = re.sub(r"\s+", " ", chunk.text).strip()
        if len(text) > settings.organize_excerpt_chars:
            text = text[:settings.organize_excerpt_chars].rstrip() + " …"
        label = _excerpt_label(db, chunk)
        excerpts.append(f"[{n}] ({label}) {text}" if label else f"[{n}] {text}")

    tasks = [
        "1. Assign every excerpt to exactly one group. Prefer an existing group when the excerpt belongs to the same "
        "learning area, and then use its name exactly. Otherwise create a new group with a short, specific name "
        "(2-6 words) that describes the subject. Never use generic names such as \"General\", \"Misc\", "
        "\"Introduction\" or the lecture/source title.",
        "2. Excerpts about the same subject belong in the same group even when they use different wording.",
        "3. Describe each group you create in `new_groups` (one sentence and a few keywords).",
    ]
    if with_concepts:
        tasks += [
            "4. For each excerpt, list the 1-4 key concepts it teaches. When a concept is already listed under an "
            "existing group, reuse that name exactly, even if the excerpt uses a synonym, abbreviation or a different "
            "spelling. Skip filler such as greetings or course logistics (use an empty list).",
            "5. Describe every concept that is not already listed in `new_concepts` (one sentence, plus its aliases: "
            "synonyms and abbreviations).",
        ]
    return "\n".join([
        "You are organizing a student's learning material into topic groups.",
        f"The numbered excerpts below come from the source \"{src.name}\".",
        f"Source hints (weak signals; trust the content over them): {hint or 'none'}",
        "",
        "EXISTING GROUPS:",
        *(group_lines or ["- none yet"]),
        "",
        "TASKS:",
        *tasks,
        "",
        "EXCERPTS:",
        *excerpts,
    ])


def _find_or_create_group(db: Session, index: dict[str, Group], name: str, meta: dict | None) -> Group:
    key = normalize_name(name)
    group = index.get(key)
    meta = meta if isinstance(meta, dict) else {}
    if group is None:
        keywords = meta.get("keywords") if isinstance(meta.get("keywords"), list) else []
        group = Group(name=name.strip()[:255], description=str(meta.get("description") or "")[:2000] or None,
                      keywords_json=json.dumps([str(k) for k in keywords][:20], ensure_ascii=False) if keywords else None)
        db.add(group)
        db.flush()
        index[key] = group
    elif meta.get("description") and group.description in (None, "", LEGACY_GROUP_DESCRIPTION):
        group.description = str(meta["description"])[:2000]
    return group


def _fallback_group(db: Session, index: dict[str, Group], embedding: list[float], previous: Group | None,
                    chunk: Chunk, src: Source) -> tuple[Group, float]:
    """Used when the model skipped an excerpt (or in offline local mode)."""
    near = nearest_groups(db, embedding, 1, settings.group_similarity_threshold)
    if near:
        return near[0][0], max(0.0, min(1.0, near[0][1]))
    if previous is not None:
        return previous, 0.0
    doc = db.get(Document, chunk.document_id)
    name = (doc.title if doc and doc.title else src.name) or "Uncategorized material"
    return _find_or_create_group(db, index, name, {"description": f"Material from {name}, not yet categorized by an AI."}), 0.0


def _update_centroids(db: Session, added: dict[int, list[list[float]]], batch_ids: list[int]) -> None:
    for group_id, embeddings in added.items():
        group = db.get(Group, group_id)
        before = db.execute(
            select(func.count(Chunk.id)).where(Chunk.group_id == group_id, Chunk.classification_status == "done",
                                               Chunk.id.not_in(batch_ids))
        ).scalar_one()
        new = np.asarray(embeddings, dtype=float)
        old = _centroid(group)
        if old and before and len(old) == new.shape[1]:
            centroid = (np.asarray(old) * before + new.sum(axis=0)) / (before + len(new))
        else:
            centroid = new.mean(axis=0)
        group.centroid_json = json.dumps(centroid.tolist())


def apply_organize(db: Session, src: Source, chunks: list[Chunk], embeddings: list[list[float]], result: dict) -> int:
    """Apply a (possibly partial or malformed) organize answer. Returns how many excerpts the model placed."""
    result = result if isinstance(result, dict) else {}
    groups_index = {normalize_name(g.name): g for g in db.execute(select(Group)).scalars().all()}
    group_meta = {normalize_name(g.get("name")): g for g in result.get("new_groups") or [] if isinstance(g, dict)}
    concept_meta = {normalize_name(c.get("name")): c for c in result.get("new_concepts") or [] if isinstance(c, dict)}
    concepts = ConceptIndex(db)

    by_excerpt: dict[int, dict] = {}
    for item in result.get("assignments") or []:
        if not isinstance(item, dict):
            continue
        try:
            n = int(item.get("excerpt"))
        except (TypeError, ValueError):
            continue
        if 1 <= n <= len(chunks) and normalize_name(str(item.get("group") or "")):
            by_excerpt.setdefault(n, item)

    added: dict[int, list[list[float]]] = defaultdict(list)
    previous: Group | None = None
    for n, (chunk, emb) in enumerate(zip(chunks, embeddings), 1):
        item = by_excerpt.get(n)
        if item:
            name = str(item["group"]).strip()
            group = _find_or_create_group(db, groups_index, name, group_meta.get(normalize_name(name)))
            confidence = 1.0
        else:
            group, confidence = _fallback_group(db, groups_index, emb, previous, chunk, src)
        chunk.group_id = group.id
        chunk.group_confidence = confidence
        chunk.classification_status = "done"
        chunk.cluster_status = "done"
        added[group.id].append(emb)
        previous = group
        if item and settings.concept_extraction:
            names = item.get("concepts") if isinstance(item.get("concepts"), list) else []
            for name in names[:6]:
                concept = concepts.resolve(str(name), concept_meta.get(normalize_name(str(name))), group)
                if concept is None:
                    continue
                exists = db.execute(select(Evidence).where(Evidence.concept_id == concept.id, Evidence.chunk_id == chunk.id)).scalar_one_or_none()
                if not exists:
                    db.add(Evidence(concept_id=concept.id, chunk_id=chunk.id, score=1.0))
                    db.flush()
    db.flush()
    _update_centroids(db, added, [c.id for c in chunks])
    return len(by_excerpt)


def organize_batch(db: Session, ai: AIProvider, src: Source, chunks: list[Chunk], hint: str | None) -> int:
    embeddings = [json.loads(c.embedding_json) for c in chunks]
    groups = candidate_groups(db, embeddings, settings.organize_group_limit)
    prompt = build_organize_prompt(db, src, hint, chunks, groups, settings.concept_extraction)
    first, last = chunks[0].id, chunks[-1].id
    result = ai.generate_json(
        prompt, task="organize", schema=organize_schema(settings.concept_extraction),
        key=f"source{src.id}:chunks{first}-{last}",
        label=f"Organize '{src.name}' ({len(chunks)} excerpts, chunks {first}-{last})",
    )
    return apply_organize(db, src, chunks, embeddings, result)
