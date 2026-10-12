"""Where a source belongs: the learner's choice of space, checked against what the content is.

The learner's choice is the source's primary place and is never changed without them. Separately,
each source is compared with the most similar sources filed in its own space and in other spaces,
using the same course-identity judgement the study plan uses (one fast AI call for the pairs not
judged yet, cached per pair). From those judgements LearnOS suggests moving the source, showing it
in both spaces, or adding an unfiled source to a space; the learner decides. A space is defined by
the sources filed in it; sources only shown there are guests and are never evidence. Placement only
organizes what is shown together: search can still reach every source.
"""
from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Course, MaintenanceJob, Source, SourceRelationship
from ..providers.base import AIProvider, ProviderAuthenticationError, ProviderRateLimitError
from .learning_paths import _classify_relationships, _digest, _is_unclassified_default, _profile
from .similarity import centroid, similarity
from .spaces import memberships

# How strongly a judged relationship says two sources belong in one space.
BELONGS = {"SAME_COURSE": 1.0, "SAME_LEARNING_PATH": 0.85, "COMPLEMENTARY_RESOURCE": 0.7}
RELATION_WORDS = {"SAME_COURSE": "the same course as", "SAME_LEARNING_PATH": "on the same learning path as",
                  "COMPLEMENTARY_RESOURCE": "a companion resource to"}
FIT = 0.6         # belonging weight x the judgement's confidence
CANDIDATES = 4    # sources compared on each side: filed in the source's own space, and filed elsewhere
PLACEMENT_TASK = "check_placement"


def _most_similar(db: Session, source: Source, others: list[Source], limit: int) -> list[Source]:
    if len(others) <= limit:
        return others
    mine = centroid(db, source)
    scored = []
    for other in others:
        score = similarity(mine, centroid(db, other))
        scored.append((score if score is not None else -2.0, other.id, other))
    return [other for _, _, other in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]


def check_placement(db: Session, ai: AIProvider, source_id: int) -> list[str]:
    """Judge the pairs that decide where this source belongs and are not judged yet: at most one AI call.

    When the AI refuses (quota, rate limit, key) the error propagates so the run can stop asking; any
    other failure leaves the pairs unjudged, to be asked again next time."""
    source = db.get(Source, source_id)
    if source is None or source.status != "done":
        return []
    others = db.execute(select(Source).where(Source.id != source.id, Source.status == "done").order_by(Source.id)).scalars().all()
    primary = source.course_id
    skip = kept_out_of(source) | memberships(db).get(source.id, set())  # refused, or already filed or shown there
    # Only sources filed in a space can say whether this one belongs there.
    near = [other for other in others if primary is not None and other.course_id == primary]
    far = [other for other in others if other.course_id is not None and other.course_id not in skip]
    chosen = _most_similar(db, source, near, CANDIDATES) + _most_similar(db, source, far, CANDIDATES)
    if not chosen:
        return []
    items = [{"id": s.id, "name": s.name, "uri": s.uri, "user_space_id": s.course_id, "profile": _profile(s)} for s in [source, *chosen]]
    warnings: list[str] = []
    _classify_relationships(db, ai, items, warnings, candidates={tuple(sorted((source.id, other.id))) for other in chosen},
                            reraise=(ProviderRateLimitError, ProviderAuthenticationError))
    return warnings


def request_placement_check(db: Session, source_ids) -> None:
    """Queue placement checks to run in the background; one waiting job collects them."""
    source_ids = [sid for sid in dict.fromkeys(source_ids) if sid is not None]
    if not source_ids:
        return
    job = db.execute(select(MaintenanceJob).where(MaintenanceJob.task == PLACEMENT_TASK, MaintenanceJob.status == "queued")
                     .order_by(MaintenanceJob.id)).scalars().first()
    items = [{"kind": "placement", "id": sid} for sid in source_ids]
    if job is None:
        db.add(MaintenanceJob(task=PLACEMENT_TASK, status="queued", progress_total=len(items), items_json=json.dumps(items)))
    else:
        queued = json.loads(job.items_json or "[]")
        waiting = {item.get("id") for item in queued[job.progress_current:] if isinstance(item, dict)}
        queued += [item for item in items if item["id"] not in waiting]
        job.items_json, job.progress_total = json.dumps(queued), len(queued)
    db.flush()


# --------------------------------------------------------------------------- the learner's answers

def _choices(source: Source) -> dict:
    try:
        data = json.loads(source.placement_json or "{}")
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def kept_out_of(source: Source) -> set[int]:
    return {int(c) for c in _choices(source).get("kept_out_of", []) if str(c).isdigit()}


def keep_out(source: Source, course_id: int) -> None:
    """Stop suggesting this space for the source."""
    data = _choices(source)
    data["kept_out_of"] = sorted(kept_out_of(source) | {course_id})
    source.placement_json = json.dumps(data)


def keep_here(source: Source, declined: int) -> None:
    """The learner keeps the source where it is: no more suggestions to move it while it is filed there."""
    keep_out(source, declined)
    if source.course_id is not None:
        data = _choices(source)
        data["kept_in"] = source.course_id
        source.placement_json = json.dumps(data)


def _kept_in(source: Source) -> int | None:
    value = _choices(source).get("kept_in")
    return int(value) if str(value).isdigit() else None


# --------------------------------------------------------------------------- suggestions

def suggestions(db: Session, sources: list[Source]) -> dict[int, dict | None]:
    """At most one suggestion per source, from judgements already made. Never calls the AI."""
    done = db.execute(select(Source).where(Source.status == "done")).scalars().all()
    by_id = {source.id: source for source in done}
    wanted = [source.id for source in sources if source.id in by_id]
    if not wanted:
        return {}
    hashes = {source.id: _digest(_profile(source)) for source in done}
    judged: dict[int, dict[int, SourceRelationship]] = {}
    # Every judgement, not only the listed sources': whether a related source belongs in its own space matters too.
    for row in db.execute(select(SourceRelationship)).scalars().all():
        if row.source_a_id not in by_id or row.source_b_id not in by_id or _is_unclassified_default(row):
            continue
        if (row.source_a_profile_hash, row.source_b_profile_hash) != (hashes[row.source_a_id], hashes[row.source_b_id]):
            continue  # judged before one of the two profiles changed
        judged.setdefault(row.source_a_id, {})[row.source_b_id] = row
        judged.setdefault(row.source_b_id, {})[row.source_a_id] = row
    filed: dict[int, set[int]] = {}  # what defines each space: its ready sources filed there
    for source in done:
        if source.course_id is not None:
            filed.setdefault(source.course_id, set()).add(source.id)
    spaces = memberships(db)  # every source, ready or not, filed or shown
    names = dict(db.execute(select(Course.id, Course.name)).all())

    def score(row: SourceRelationship) -> float:
        return BELONGS.get(row.relation_type, 0.0) * float(row.confidence or 0.0)

    def best_fit(source_id: int, others) -> float:
        rows = [judged.get(source_id, {}).get(other) for other in others]
        return max((score(row) for row in rows if row is not None), default=0.0)

    def standing(other: int, course_id: int, ignoring: int) -> str:
        """Does `other` itself belong in the space? Only then is it evidence for joining it."""
        neighbours = filed.get(course_id, set()) - {other, ignoring}
        if not neighbours:
            return "alone"
        if not any(n in judged.get(other, {}) for n in neighbours):
            return "unknown"
        return "belongs" if best_fit(other, neighbours) >= FIT else "misfit"

    out: dict[int, dict | None] = {}
    for source_id in wanted:
        source = by_id[source_id]
        out[source_id] = None
        primary = source.course_id
        neighbours = filed.get(primary, set()) - {source_id} if primary is not None else set()
        if neighbours and not any(other in judged.get(source_id, {}) for other in neighbours):
            continue  # how it fits its own space is not judged yet: no basis for a comparison
        own_fit = best_fit(source_id, neighbours)
        shown, refused = spaces.get(source_id, set()), kept_out_of(source)
        options = []
        for course_id in sorted(filed):
            if course_id in shown or course_id in refused:
                continue
            for other in sorted(filed[course_id] - {source_id}):
                row = judged.get(source_id, {}).get(other)
                if row is None or score(row) < FIT:
                    continue
                place = standing(other, course_id, source_id)
                if place in {"misfit", "unknown"}:
                    continue  # a related source that is itself misfiled (or not judged yet) says nothing about the space
                # Between a lone source and one with neighbours, only one of them is ever asked to move:
                if place == "alone" and neighbours and score(row) <= own_fit:
                    continue  # it fits its own space at least as well as the lone source: the lone one joins it
                if place == "belongs" and primary is not None and not neighbours and \
                        best_fit(other, filed[course_id] - {other, source_id}) < score(row) \
                        and primary not in kept_out_of(by_id[other]):
                    continue  # the related source belongs with this one more than in its own space (unless it refused it)
                # Two lone sources of one course: only the one in the newer space joins the other, unless that was refused.
                if place == "alone" and primary is not None and not neighbours and course_id > primary \
                        and primary not in kept_out_of(by_id[other]):
                    continue
                options.append((score(row), course_id, row))
        if not options:
            continue
        _, course_id, row = max(options, key=lambda option: (option[0], -option[1]))
        if primary is None:
            kind = "add"
        else:
            kind = "also" if neighbours and own_fit >= FIT else "move"
            if kind == "move" and _kept_in(source) == primary:
                continue  # the learner chose to keep it here
        other = by_id[row.source_b_id if row.source_a_id == source_id else row.source_a_id]
        out[source_id] = {
            "kind": kind, "course_id": course_id, "course_name": names.get(course_id),
            "relation": row.relation_type, "confidence": round(float(row.confidence or 0.0), 2),
            "related_source": {"id": other.id, "name": other.name},
            "reason": f"Looks like {RELATION_WORDS.get(row.relation_type, 'related to')} “{other.name}”. {row.rationale or ''}".strip(),
            # Moving would leave its current space with no sources at all (ready, processing or shown there).
            "leaves_space_empty": kind == "move" and not any(sid != source_id and primary in ids for sid, ids in spaces.items()),
        }
    # Decided per space, not per pair: when a lone source is asked to join a space, that space's sources are not
    # also asked to go to the lone source's space (through a third source, the pairs alone can disagree).
    joining = {(source.course_id, suggestion["course_id"]) for source_id, suggestion in out.items() if suggestion
               for source in [by_id[source_id]] if source.course_id is not None and not filed.get(source.course_id, set()) - {source_id}}
    for source_id, suggestion in out.items():
        if suggestion and (suggestion["course_id"], by_id[source_id].course_id) in joining:
            out[source_id] = None
    return out
