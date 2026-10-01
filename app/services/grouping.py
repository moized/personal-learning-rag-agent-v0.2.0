import json
import math
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..models import Chunk, Group, Concept, Evidence
from ..providers.base import AIProvider
from ..config import settings


def cosine(a: list[float], b: list[float]) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return 0.0 if den == 0 else float(np.dot(x, y) / den)


def nearest_groups(db: Session, embedding: list[float], limit: int, threshold: float) -> list[tuple[Group, float]]:
    rows = db.execute(select(Group)).scalars().all()
    scored: list[tuple[Group, float]] = []
    for group in rows:
        if not group.centroid_json:
            continue
        try:
            score = cosine(embedding, json.loads(group.centroid_json))
        except (TypeError, json.JSONDecodeError):
            continue
        if score >= threshold:
            scored.append((group, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


def _hint_text(hint: str | None) -> str:
    return (hint or "none")[:1500]


def choose_group(
    db: Session,
    ai: AIProvider,
    chunk: Chunk,
    embedding: list[float],
    hint: str | None,
    threshold: float,
) -> tuple[Group, float]:
    candidates = nearest_groups(db, embedding, settings.group_candidates, threshold)

    if not candidates:
        name = (hint or "General").strip()[:255]
        group = db.execute(select(Group).where(Group.name == name)).scalar_one_or_none()
        if not group:
            group = Group(name=name, description="Automatically created semantic group", centroid_json=json.dumps(embedding))
            db.add(group)
            db.flush()
        return group, 0.5

    # Very clear candidate: avoid paying for an LLM call.
    if len(candidates) == 1 or not settings.group_llm_validation:
        return candidates[0][0], max(0.0, min(1.0, candidates[0][1]))
    margin = candidates[0][1] - candidates[1][1]
    if margin >= 0.08 and candidates[0][1] >= threshold + 0.03:
        return candidates[0][0], max(0.0, min(1.0, candidates[0][1]))

    candidate_text = "\n".join(
        f"{g.id}. {g.name}: {g.description or ''} | keywords={g.keywords_json or '[]'}"
        for g, _ in candidates
    )
    prompt = f"""You are routing a learning chunk into an existing semantic knowledge group.
Prefer an existing group only when it is genuinely about the same learning area. Use the source hints as weak context, not truth.
Chunk summary: {chunk.summary or chunk.text[:900]}
Source hints: {_hint_text(hint)}
Candidate groups:
{candidate_text}
Return JSON with the selected numeric group_id and confidence from 0 to 1.
"""
    schema = {
        "type": "object",
        "properties": {
            "group_id": {"type": "integer"},
            "confidence": {"type": "number"},
        },
        "required": ["group_id", "confidence"],
    }
    result = ai.generate_json(prompt, task="routing", schema=schema)
    selected = int(result.get("group_id", candidates[0][0].id))
    confidence = float(result.get("confidence", candidates[0][1]))
    allowed = {g.id: g for g, _ in candidates}
    group = allowed.get(selected, candidates[0][0])
    return group, max(0.0, min(1.0, confidence))


def extract_group_concepts(db: Session, ai: AIProvider, group: Group) -> int:
    """Extract concepts once per semantic group rather than once per chunk."""
    if not settings.concept_extraction:
        return 0
    rows = db.execute(
        select(Chunk).where(Chunk.group_id == group.id, Chunk.embedding_status == "done").order_by(Chunk.id).limit(24)
    ).scalars().all()
    if not rows:
        return 0
    snippets = "\n\n".join(f"CHUNK {c.id}: {(c.contextual_text or c.text)[:1400]}" for c in rows)
    prompt = f"""Build a compact concept index for the learning group below.
Merge synonyms across chunks. Keep only meaningful technical/learning concepts.
Group: {group.name}
Description: {group.description or ''}
Evidence snippets:
{snippets}
Return 5-30 canonical concepts, each with a one-sentence description and optional aliases.
"""
    schema = {
        "type": "object",
        "properties": {
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "description"],
                },
            }
        },
        "required": ["concepts"],
    }
    try:
        result = ai.generate_json(prompt, task="extraction", schema=schema)
    except Exception:
        return 0
    count = 0
    for item in result.get("concepts", []):
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        concept = db.execute(select(Concept).where(Concept.canonical_name == name)).scalar_one_or_none()
        aliases = item.get("aliases") if isinstance(item.get("aliases"), list) else []
        if not concept:
            concept = Concept(canonical_name=name, description=str(item.get("description", ""))[:2000],
                              aliases_json=json.dumps(aliases[:30], ensure_ascii=False), group_id=group.id)
            db.add(concept)
            db.flush()
            count += 1
        else:
            concept.group_id = concept.group_id or group.id
            if not concept.description:
                concept.description = str(item.get("description", ""))[:2000]
            old = json.loads(concept.aliases_json or "[]")
            concept.aliases_json = json.dumps(list(dict.fromkeys([*old, *aliases]))[:30], ensure_ascii=False)
        # Keep a few high-signal evidence links rather than linking every chunk.
        needle = name.lower()
        candidates = sorted(rows, key=lambda c: (needle in c.text.lower(), len(c.text)))
        for chunk in candidates[-3:]:
            evidence = db.execute(select(Evidence).where(Evidence.concept_id == concept.id, Evidence.chunk_id == chunk.id)).scalar_one_or_none()
            if not evidence:
                db.add(Evidence(concept_id=concept.id, chunk_id=chunk.id, score=1.0))
    return count


def merge_centroid(old: list[float] | None, new: list[float], count: int) -> list[float]:
    if old is None:
        return new
    # Running mean with the number of members BEFORE the new one.
    previous = max(count - 1, 0)
    if previous == 0:
        return new
    return ((np.asarray(old) * previous + np.asarray(new)) / count).tolist()


def extract_concepts(db: Session, ai: AIProvider, chunk: Chunk, group: Group) -> None:
    if not settings.concept_extraction:
        return
    prompt = f"""Extract the important canonical learning concepts from this chunk.
Do not create generic filler concepts. Merge synonyms mentally and return concise canonical names.
Group: {group.name}
Chunk: {(chunk.contextual_text or chunk.text)[:6000]}
Return 2-8 concepts with a one-sentence description and optional aliases.
"""
    schema = {
        "type": "object",
        "properties": {
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "description"],
                },
            }
        },
        "required": ["concepts"],
    }
    result = ai.generate_json(prompt, task="extraction", schema=schema)
    for item in result.get("concepts", []):
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        concept = db.execute(select(Concept).where(Concept.canonical_name == name)).scalar_one_or_none()
        aliases = item.get("aliases") if isinstance(item.get("aliases"), list) else []
        if not concept:
            concept = Concept(
                canonical_name=name,
                description=str(item.get("description", ""))[:2000],
                aliases_json=json.dumps(aliases, ensure_ascii=False),
                group_id=group.id,
            )
            db.add(concept)
            db.flush()
        else:
            if concept.group_id is None:
                concept.group_id = group.id
            if aliases:
                old = json.loads(concept.aliases_json or "[]")
                merged = list(dict.fromkeys([*old, *aliases]))
                concept.aliases_json = json.dumps(merged[:30], ensure_ascii=False)
        evidence = db.execute(
            select(Evidence).where(Evidence.concept_id == concept.id, Evidence.chunk_id == chunk.id)
        ).scalar_one_or_none()
        if not evidence:
            db.add(Evidence(concept_id=concept.id, chunk_id=chunk.id, score=1.0))
