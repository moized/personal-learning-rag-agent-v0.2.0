"""Which passages each study topic teaches, and which earlier topics from other sources it overlaps.

Dividing shared passages. The plan links topics to sections. A source without chapters (most videos)
is a single section, so every topic the plan finds in it would get all of it, and the learner would
study the same passages under several titles. The passages of a section that several topics share are
divided between those topics instead: every passage goes to exactly one of them and each gets at least
one, so nothing is studied twice and nothing is left out. Each passage is scored against each topic in
two ways: BM25 on the topic's title and purpose (without common words, and with words cut to a short
stem so "servers" finds "server"), and the cosine similarity of its embedding to a topic vector made
from the passages that match the topic's words best (pseudo-relevance feedback). Embeddings are centred
on the section's own mean first: raw similarities of these models sit in a narrow band, and the
speaker's style is common to every passage of a source. The standardized scores are combined, and a
Viterbi pass with a cost for changing topic keeps runs of consecutive passages together, as a talk
moves from one subject to the next. A section is left whole (every topic teaches all of it, as before)
when it has fewer passages than topics, or when a topic's words appear nowhere in it, for example a
transcript in another language than the plan: then there is nothing to tell its topics apart by.

Overlap across sources. Sources that teach the same thing in their own words rarely share near-identical
passages, so overlap is judged from the key ideas LearnOS read in them: a topic that shares key ideas
with an earlier topic from another source is linked to it. Nothing is merged or dropped. The unit says
which ideas were met before and which are new, so the tutor recalls the first briefly and spends the
session on the rest.

No AI call is made. Results are kept in memory per plan build and made again when the passages or key
ideas change.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import Chunk, Concept, Evidence, Section, Source, StudyPlan, StudyPlanNode
from .plans import reading_order

SWITCH_COST = 8.0      # in standard deviations of a passage's score: an aside of a few passages does not change the topic
LEXICAL_WEIGHT = 0.5   # BM25 next to the embedding similarity
FEEDBACK_SHARE = 20    # a topic's vector is the mean of its best-matching twentieth of the passages (at least 3)
STEM = 5               # words are compared by their first letters, after a plural "s" is removed
MIN_SHARED_IDEAS = 2   # key ideas two topics must share to overlap (or all of a topic's ideas, when it has fewer)
MAX_OVERLAPS = 2       # earlier topics shown per topic
MAX_LISTED_IDEAS = 6   # key ideas shown per list
_CACHE_SIZE = 8
_K1, _B = 1.5, 0.75    # BM25, as in retrieval
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_STOPWORDS = frozenset("""a about after all also an and any are as at be been before between both but by can do does for from
has have how if in into is it its more most not of on or other our over such than that the their them then there these they
this those through to under up use used using via was what when where which while who why will with within without you your
learn understand explain introduce explore""".split())


@dataclass(frozen=True)
class Coverage:
    # Topics that share a divided section with other topics: the passages each one teaches, in reading
    # order. Topics missing here teach every passage of their sections.
    passages: dict[str, list[int]] = field(default_factory=dict)
    # Every earlier topic from another source that shares enough key ideas, best first:
    # [{"id", "title", "path", "shared": [names, most taught first]}]. Shown capped (see shown()).
    overlaps: dict[str, list[dict]] = field(default_factory=dict)
    # For a topic with overlaps: its key ideas met in no earlier topic, most taught first.
    new_ideas: dict[str, list[str]] = field(default_factory=dict)
    # Every topic's key ideas, most taught first.
    ideas: dict[str, list[str]] = field(default_factory=dict)
    # The names of the sources each topic's sections come from (as the learner named them).
    sources: dict[str, list[str]] = field(default_factory=dict)


def narrowed(overlaps: list[dict], new_ideas: list[str], taught: set[str]) -> tuple[list[dict], list[str]]:
    """The overlaps of a part of a topic: only the key ideas its own passages teach."""
    items = [{**item, "shared": [idea for idea in item["shared"] if idea in taught]} for item in overlaps]
    items = sorted((item for item in items if item["shared"]), key=lambda item: -len(item["shared"]))
    return items, [idea for idea in new_ideas if idea in taught]


def shown(overlaps: list[dict], new_ideas: list[str]) -> tuple[list[dict], list[str]]:
    """What a unit or prompt shows: the best few overlaps with their main key ideas."""
    return ([{**item, "shared": item["shared"][:MAX_LISTED_IDEAS]} for item in overlaps[:MAX_OVERLAPS]],
            new_ideas[:MAX_LISTED_IDEAS])


_divisions: dict[tuple, dict[str, list[int]]] = {}
_coverages: dict[tuple, Coverage] = {}
_guard = threading.Lock()


def _remember(cache: dict, key: tuple, value):
    with _guard:
        if len(cache) >= _CACHE_SIZE:
            cache.pop(next(iter(cache)))
        cache[key] = value
    return value


def _recall(cache: dict, key: tuple):
    with _guard:
        return cache.get(key)


def _meta(node: StudyPlanNode) -> dict:
    try:
        meta = json.loads(node.metadata_json or "{}")
    except (TypeError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _section_ids(node: StudyPlanNode) -> list[int]:
    return list(dict.fromkeys(s for s in _meta(node).get("section_ids") or [] if isinstance(s, int)))


def _plan_key(plan: StudyPlan, rows: list[StudyPlanNode]) -> tuple:
    """A plan build: its revision, when it was built, and its topics with their sections and titles."""
    layout = [(row.node_id, row.parent_node_id, row.path_id, row.node_type, row.title, row.metadata_json or "") for row in rows]
    digest = hashlib.sha256(json.dumps(layout, ensure_ascii=False).encode()).hexdigest()
    return plan.id, plan.revision, str(plan.generated_at), digest


# ------------------------------------------------------------------ dividing a shared section

def _terms(text: str) -> list[str]:
    out = []
    for word in _WORD.findall((text or "").lower()):
        if word in _STOPWORDS or (len(word) < 2 and not word.isdigit()):
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        out.append(word[:STEM])
    return out


def _lexical(chunks: list[Chunk], queries: list[str]) -> np.ndarray:
    """BM25 of every passage (rows) for every topic's words (columns)."""
    docs = [Counter(_terms(chunk.text)) for chunk in chunks]
    lengths = [sum(doc.values()) for doc in docs]
    average = max(sum(lengths) / max(len(docs), 1), 1.0)
    frequency = Counter(term for doc in docs for term in doc)
    n = len(docs)
    out = np.zeros((n, len(queries)))
    for column, query in enumerate(queries):
        for term in set(_terms(query)):
            if not frequency[term]:
                continue
            idf = math.log(1 + (n - frequency[term] + 0.5) / (frequency[term] + 0.5))
            for row, doc in enumerate(docs):
                count = doc.get(term)
                if count:
                    out[row, column] += idf * count * (_K1 + 1) / (count + _K1 * (1 - _B + _B * lengths[row] / average))
    return out


def _standardized(matrix: np.ndarray) -> np.ndarray:
    """Each topic's scores as standard deviations from its mean, so topics with more matching words do
    not take everything. Missing scores (NaN) count as average."""
    out = np.zeros_like(matrix)
    present = ~np.isnan(matrix).all(axis=0)  # a topic with no scores at all stays average everywhere
    if present.any():
        scores = matrix[:, present]
        spread = np.nanstd(scores, axis=0)
        scaled = (scores - np.nanmean(scores, axis=0)) / np.where(spread > 1e-9, spread, 1.0)
        scaled[:, spread <= 1e-9] = 0.0
        out[:, present] = np.nan_to_num(scaled, nan=0.0)
    return out


def _embeddings(chunks: list[Chunk]) -> np.ndarray | None:
    """Passage vectors centred on their mean and scaled to unit length; NaN rows for passages without one
    (or embedded by another model). None when no passage has one."""
    parsed = []
    for chunk in chunks:
        try:
            vector = json.loads(chunk.embedding_json) if chunk.embedding_json else None
        except (TypeError, ValueError):
            vector = None
        parsed.append(vector if isinstance(vector, list) and vector else None)
    sizes: dict[int, int] = {}
    for vector in parsed:
        if vector is not None:
            sizes[len(vector)] = sizes.get(len(vector), 0) + 1
    if not sizes:
        return None
    size = max(sizes, key=lambda length: (sizes[length], length))
    matrix = np.full((len(chunks), size), np.nan)
    for row, vector in enumerate(parsed):
        if vector is not None and len(vector) == size:
            matrix[row] = vector
    valid = ~np.isnan(matrix[:, 0])
    matrix[valid] -= matrix[valid].mean(axis=0)
    norms = np.linalg.norm(matrix[valid], axis=1, keepdims=True)
    matrix[valid] /= np.where(norms > 1e-12, norms, 1.0)
    return matrix


def _viterbi(score: np.ndarray, cost: float) -> np.ndarray:
    """The topic for each passage that maximizes the total score minus `cost` per change of topic."""
    n, k = score.shape
    best = score[0].copy()
    back = np.zeros((n, k), dtype=np.int64)
    states = np.arange(k)
    for i in range(1, n):
        leader = int(np.argmax(best))
        switched = best[leader] - cost
        stay = best >= switched
        back[i] = np.where(stay, states, leader)
        best = np.where(stay, best, switched) + score[i]
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmax(best))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def _give_every_topic_a_share(path: np.ndarray, score: np.ndarray) -> None:
    """A topic the plan found here and the Viterbi pass left empty gets a short run where it stands out
    most from the others, taken only from topics that keep passages of their own. Needs n >= k."""
    n, k = score.shape
    width = max(1, n // (4 * k))
    counts = np.bincount(path, minlength=k)
    for topic in range(k):
        if counts[topic]:
            continue
        margin = score[:, topic] - np.delete(score, topic, axis=1).max(axis=1)
        eligible = counts[path] > 1  # with n >= k and an empty topic, some topic always has two or more
        centre = int(np.argmax(np.where(eligible, margin, -np.inf)))
        donor = int(path[centre])
        run, low, high = [centre], centre, centre
        while len(run) < width and counts[donor] - len(run) > 1:
            sides = [i for i in (low - 1, high + 1) if 0 <= i < n and path[i] == donor]
            if not sides:
                break
            step = max(sides, key=lambda i: margin[i])
            run.append(step)
            low, high = min(low, step), max(high, step)
        path[run] = topic
        counts[donor] -= len(run)
        counts[topic] = len(run)


def divide(chunks: list[Chunk], queries: list[str]) -> list[int] | None:
    """For passages in reading order, the index of the topic (described by its query) each one goes to;
    None when the section is better left whole (see the module docstring)."""
    n, k = len(chunks), len(queries)
    if k < 2 or n < k:
        return None
    lexical = _lexical(chunks, queries)
    if not (lexical.max(axis=0) > 0).all():
        return None  # a topic whose words appear nowhere here: nothing to tell it apart by
    score = _standardized(lexical)
    vectors = _embeddings(chunks)
    if vectors is not None:
        valid = ~np.isnan(vectors[:, 0])
        semantic = np.full((n, k), np.nan)
        feedback = max(3, n // FEEDBACK_SHARE)
        for topic in range(k):
            ranked = [i for i in np.argsort(-lexical[:, topic], kind="stable") if lexical[i, topic] > 0 and valid[i]][:feedback]
            if not ranked:
                continue
            anchor = vectors[ranked].mean(axis=0)
            norm = float(np.linalg.norm(anchor))
            if norm > 1e-12:
                semantic[valid, topic] = vectors[valid] @ (anchor / norm)
        if not np.isnan(semantic).all():
            score = _standardized(semantic) + LEXICAL_WEIGHT * score
    path = _viterbi(score, SWITCH_COST)
    _give_every_topic_a_share(path, score)
    return [int(topic) for topic in path]


def _query(node: StudyPlanNode) -> str:
    purpose = str(_meta(node).get("purpose") or "")
    return f"{node.title}. {purpose}".strip()


def _divisions_for(db: Session, plan_key: tuple, study: list[StudyPlanNode], sections: dict[int, Section]) -> dict[str, list[int]]:
    users: dict[int, list[StudyPlanNode]] = {}
    for node in study:
        for section_id in _section_ids(node):
            if section_id in sections:
                users.setdefault(section_id, []).append(node)
    shared = sorted(section_id for section_id, nodes in users.items() if len(nodes) > 1)
    if not shared:
        return {}
    shape = db.execute(select(func.count(Chunk.id), func.max(Chunk.id), func.count(Chunk.embedding_json))
                       .where(Chunk.section_id.in_(shared), Chunk.chunk_kind == "leaf")).one()
    key = (*plan_key, tuple(shape))
    known = _recall(_divisions, key)
    if known is not None:
        return known
    share: dict[tuple[str, int], list[int]] = {}
    divided_sections: set[int] = set()
    for section_id in shared:
        chunks = db.execute(select(Chunk).where(Chunk.section_id == section_id, Chunk.chunk_kind == "leaf")
                            .order_by(Chunk.document_id, Chunk.chunk_index)).scalars().all()
        nodes = users[section_id]
        path = divide(chunks, [_query(node) for node in nodes])
        if path is None:
            continue
        divided_sections.add(section_id)
        for chunk, topic in zip(chunks, path):
            share.setdefault((nodes[topic].node_id, section_id), []).append(chunk.id)
    divided = {node.node_id for section_id in divided_sections for node in users[section_id]}
    order = sorted(sections.values(), key=lambda s: (s.source_id, s.position, s.id))
    passages: dict[str, list[int]] = {}
    for node in study:
        if node.node_id not in divided:
            continue
        mine = set(_section_ids(node))
        ids: list[int] = []
        for section in order:
            if section.id not in mine:
                continue
            if section.id in divided_sections:
                ids.extend(share.get((node.node_id, section.id), []))
            else:
                ids.extend(db.execute(select(Chunk.id).where(Chunk.section_id == section.id, Chunk.chunk_kind == "leaf")
                                      .order_by(Chunk.document_id, Chunk.chunk_index)).scalars().all())
        passages[node.node_id] = ids
    return _remember(_divisions, key, passages)


# ------------------------------------------------------------------------ overlap

def plan_coverage(db: Session, plan: StudyPlan | None) -> Coverage:
    if plan is None:
        return Coverage()
    rows = db.execute(select(StudyPlanNode).where(StudyPlanNode.plan_id == plan.id).order_by(StudyPlanNode.id)).scalars().all()
    ordered = reading_order(rows)
    study = [node for node in ordered if node.node_type == "study"]
    wanted = {section_id for node in study for section_id in _section_ids(node)}
    sections = {s.id: s for s in db.execute(select(Section).where(Section.id.in_(wanted))).scalars().all()} if wanted else {}
    if not study or not sections:
        return Coverage()
    plan_key = _plan_key(plan, rows)
    passages = _divisions_for(db, plan_key, study, sections)

    leaf_ids = select(Chunk.id).where(Chunk.section_id.in_(list(sections)), Chunk.chunk_kind == "leaf")
    ideas_shape = db.execute(select(func.count(Evidence.id), func.max(Evidence.id)).where(Evidence.chunk_id.in_(leaf_ids))).one()
    key = (*plan_key, tuple(ideas_shape), json.dumps(passages, sort_keys=True))
    known = _recall(_coverages, key)
    if known is not None:
        return known

    section_of = dict(db.execute(select(Chunk.id, Chunk.section_id).where(Chunk.id.in_(leaf_ids))).all())
    chunks_of_section: dict[int, list[int]] = {}
    for chunk_id, section_id in section_of.items():
        chunks_of_section.setdefault(section_id, []).append(chunk_id)
    ideas_of_chunk: dict[int, list[int]] = {}
    for chunk_id, concept_id in db.execute(select(Evidence.chunk_id, Evidence.concept_id).where(Evidence.chunk_id.in_(leaf_ids))).all():
        ideas_of_chunk.setdefault(chunk_id, []).append(concept_id)
    names = dict(db.execute(select(Concept.id, Concept.canonical_name)
                            .where(Concept.id.in_({c for ids in ideas_of_chunk.values() for c in ids}))).all())
    paths = {node.node_id: node.title for node in rows if node.node_id == node.path_id}
    source_names = dict(db.execute(select(Source.id, Source.name).where(Source.id.in_({s.source_id for s in sections.values()}))).all())
    named = {node.node_id: list(dict.fromkeys(source_names[sections[s].source_id] for s in _section_ids(node)
                                              if s in sections and sections[s].source_id in source_names)) for node in study}

    weight: dict[str, dict[int, int]] = {}   # topic -> key idea -> passages that teach it
    sources: dict[str, set[int]] = {}
    for node in study:
        own = passages.get(node.node_id)
        if own is None:
            own = [chunk_id for section_id in _section_ids(node) for chunk_id in chunks_of_section.get(section_id, [])]
        counts: dict[int, int] = {}
        for chunk_id in own:
            for concept_id in ideas_of_chunk.get(chunk_id, []):
                counts[concept_id] = counts.get(concept_id, 0) + 1
        weight[node.node_id] = counts
        sources[node.node_id] = {sections[s].source_id for s in _section_ids(node) if s in sections}

    def listed(node_id: str, concept_ids: set[int]) -> list[str]:
        ranked = sorted(concept_ids, key=lambda c: (-weight[node_id].get(c, 0), names.get(c, "")))
        return [names[c] for c in ranked if c in names]

    overlaps: dict[str, list[dict]] = {}
    new_ideas: dict[str, list[str]] = {}
    for index, node in enumerate(study):
        mine = set(weight[node.node_id])
        if not mine:
            continue
        needed = min(MIN_SHARED_IDEAS, len(mine))
        found = []
        for position, earlier in enumerate(study[:index]):
            if sources[earlier.node_id] & sources[node.node_id]:
                continue  # topics of one source follow each other; only other sources repeat
            shared = mine & set(weight[earlier.node_id])
            if len(shared) >= needed:
                found.append((-len(shared), position, earlier, shared))
        if not found:
            continue
        found.sort(key=lambda item: (item[0], item[1]))
        overlaps[node.node_id] = [{"id": earlier.node_id, "title": earlier.title, "path": paths.get(earlier.path_id, ""),
                                   "source": ", ".join(named[earlier.node_id]), "shared": listed(node.node_id, shared)}
                                  for _, _, earlier, shared in found]
        seen = {c for earlier in study[:index] for c in weight[earlier.node_id]}
        new_ideas[node.node_id] = listed(node.node_id, mine - seen)
    ideas = {node.node_id: listed(node.node_id, set(weight[node.node_id])) for node in study}
    return _remember(_coverages, key, Coverage(passages=passages, overlaps=overlaps, new_ideas=new_ideas, ideas=ideas, sources=named))
