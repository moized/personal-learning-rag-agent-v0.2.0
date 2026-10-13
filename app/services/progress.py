"""Study progress: which plan topics and parts the learner has studied.

Progress is saved per topic id (titles can collide and change), with rows from before topics had
ids matched by title. A topic divided into parts counts as studied once all its parts at one session
size are; un-ticking a part re-opens the topics above it unless they are still complete at another size.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import PlanArrangement, StudyPlan, StudyPlanNode, StudyPosition, StudyProgress, TopicDetail


def topic_key(title: str) -> str:
    return " ".join(title.casefold().split())


def is_plan_topic(db: Session, scope: str, node_id: str) -> bool:
    return db.execute(select(StudyPlanNode.id).join(StudyPlan, StudyPlanNode.plan_id == StudyPlan.id)
                      .where(StudyPlan.scope_key == scope, StudyPlanNode.node_id == node_id)).first() is not None


def save_progress(db: Session, scope: str, topic: str, topic_id: str | None, completed: bool, *, claim_legacy: bool = True) -> None:
    """Tick or un-tick one topic or part. `claim_legacy` lets a plan topic take over a row saved by
    title before topics had ids; a tick the learner did not make (a roll-up) never does, because
    another topic with the same title may rely on that row."""
    key = topic_key(topic)
    row = None
    if topic_id:  # the topic id identifies progress across rewordings; titles can collide
        row = db.execute(select(StudyProgress).where(StudyProgress.scope_key == scope, StudyProgress.node_id == topic_id)
                         .order_by(StudyProgress.id)).scalars().first()
    title_row = db.execute(select(StudyProgress).where(StudyProgress.scope_key == scope, StudyProgress.topic_key == key)).scalar_one_or_none()
    # A row found by title belongs to this topic only if it has the same id, or no id yet: rows saved
    # before topics had ids belong to plan topics, never to a part of one that has the same title.
    title_is_ours = title_row is not None and (
        not topic_id or title_row.node_id == topic_id
        or (claim_legacy and title_row.node_id is None and is_plan_topic(db, scope, topic_id)))
    if row is None and title_is_ours:
        row = title_row
    if completed:
        if row is None:
            # The title may already be used by another topic's progress: keep both, with distinct keys.
            stored_key = key if title_row is None else f"{key}#{topic_id}"[:500]
            db.add(StudyProgress(scope_key=scope, topic_key=stored_key, topic_label=topic.strip(), node_id=topic_id, completed=True))
        else:
            row.topic_label = topic.strip()
            row.node_id = row.node_id or topic_id
            row.completed = True
    elif row is not None:
        db.delete(row)
    db.flush()


def _done_ids(db: Session, scope: str, ids: set[str]) -> set[str]:
    if not ids:
        return set()
    return set(db.execute(select(StudyProgress.node_id).where(
        StudyProgress.scope_key == scope, StudyProgress.completed.is_(True), StudyProgress.node_id.in_(ids))).scalars().all())


def _parts_by_size(db: Session, scope: str, parent_id: str) -> dict[int, set[str]]:
    sizes: dict[int, set[str]] = {}
    for node_id, budget in db.execute(select(TopicDetail.node_id, TopicDetail.budget_tokens).where(
            TopicDetail.scope_key == scope, TopicDetail.parent_node_id == parent_id)).all():
        sizes.setdefault(budget, set()).add(node_id)
    return sizes


def _title(db: Session, scope: str, node_id: str, budget: int) -> str | None:
    title = db.execute(select(TopicDetail.title).where(
        TopicDetail.scope_key == scope, TopicDetail.node_id == node_id, TopicDetail.budget_tokens == budget)).scalar()
    if title is None:  # a topic of the plan itself
        title = db.execute(select(StudyPlanNode.title).join(StudyPlan, StudyPlanNode.plan_id == StudyPlan.id).where(
            StudyPlan.scope_key == scope, StudyPlanNode.node_id == node_id)).scalar()
    return title


def roll_up(db: Session, scope: str, node_id: str, completed: bool) -> None:
    """After a part was ticked or un-ticked, update the topics above it."""
    for _ in range(10):  # parts are at most a few levels deep
        part = db.execute(select(TopicDetail).where(TopicDetail.scope_key == scope, TopicDetail.node_id == node_id)
                          .order_by(TopicDetail.id)).scalars().first()
        if part is None or not part.parent_node_id:
            return
        sizes = _parts_by_size(db, scope, part.parent_node_id)
        siblings = sizes.get(part.budget_tokens, set())
        if completed and siblings - _done_ids(db, scope, siblings):
            return  # other parts at this size are still to study
        if not completed and any(parts and parts <= _done_ids(db, scope, parts)
                                 for budget, parts in sizes.items() if budget != part.budget_tokens):
            return  # still fully studied at another session size
        title = _title(db, scope, part.parent_node_id, part.budget_tokens)
        if title is None:
            return
        save_progress(db, scope, title, part.parent_node_id, completed, claim_legacy=False)
        node_id = part.parent_node_id


def reopen(db: Session, scope: str, node_id: str, budget: int) -> None:
    """A topic gained material it was not studied with: it and the topics above it are no longer done."""
    title = _title(db, scope, node_id, budget)
    if title is None or not _done_ids(db, scope, {node_id}):
        return
    save_progress(db, scope, title, node_id, False, claim_legacy=False)
    roll_up(db, scope, node_id, False)


def plan_topic_of(db: Session, scope: str) -> dict[str, str]:
    """For every part in the scope, the plan topic it belongs to (any session size)."""
    parents = dict(db.execute(select(TopicDetail.node_id, TopicDetail.parent_node_id).where(
        TopicDetail.scope_key == scope, TopicDetail.parent_node_id.is_not(None))).all())
    roots: dict[str, str] = {}
    for node_id in parents:
        current, seen = node_id, set()
        while current in parents and current not in seen:
            seen.add(current)
            current = parents[current]
        roots[node_id] = current
    return roots


def remember_position(db: Session, scope: str, node_id: str, budget: int, title: str) -> None:
    """The learner opened this study session: it is where they continue next time, even unfinished."""
    parents = dict(db.execute(select(TopicDetail.node_id, TopicDetail.parent_node_id).where(
        TopicDetail.scope_key == scope, TopicDetail.budget_tokens == budget)).all())
    trail = [node_id]
    while parents.get(trail[0]) and len(trail) < 10:
        trail.insert(0, parents[trail[0]])
    row = db.execute(select(StudyPosition).where(StudyPosition.scope_key == scope)).scalar_one_or_none()
    if row is None:
        row = StudyPosition(scope_key=scope)
        db.add(row)
    row.node_id, row.trail_json, row.budget_tokens, row.title = node_id, json.dumps(trail), budget, title[:500]
    row.updated_at = datetime.now(UTC)  # every visit counts, also of the same session again
    db.flush()


def _is_session(db: Session, scope: str, node_id: str) -> bool:
    """A session of topics the learner's arrangement studies together."""
    from .arrangement import GROUP_PREFIX, sessions  # it saves progress through this module
    if not node_id.startswith(GROUP_PREFIX):
        return False
    plan = db.execute(select(StudyPlan).where(StudyPlan.scope_key == scope)).scalar_one_or_none()
    return any(session.id == node_id for session in sessions(db, plan))


def _utc(moment):
    """Stored times are UTC without a zone; ones not yet read back from the database carry it."""
    return moment.replace(tzinfo=None) if moment is not None and moment.tzinfo is not None else moment


def current_position(db: Session, scope: str) -> dict | None:
    """The session the learner opened last, while it is still in the plan and not yet studied. A session
    opened before the learner applied an arrangement belongs to the plan's earlier order: the arranged
    plan starts from its first session not yet studied instead."""
    row = db.execute(select(StudyPosition).where(StudyPosition.scope_key == scope)).scalar_one_or_none()
    if row is None:
        return None
    arranged = db.execute(select(PlanArrangement).where(PlanArrangement.scope_key == scope)).scalar_one_or_none()
    applied = _utc(arranged.applied_at or arranged.created_at) if arranged is not None else None
    if applied is not None and row.updated_at is not None and _utc(row.updated_at) < applied:
        return None
    try:
        trail = [str(node_id) for node_id in json.loads(row.trail_json or "[]")]
    except (TypeError, ValueError):
        return None
    if not trail or trail[-1] != row.node_id or not (is_plan_topic(db, scope, trail[0]) or _is_session(db, scope, trail[0])):
        return None
    parts = set(trail[1:])
    if parts and parts - set(db.execute(select(TopicDetail.node_id).where(
            TopicDetail.scope_key == scope, TopicDetail.budget_tokens == row.budget_tokens,
            TopicDetail.node_id.in_(parts))).scalars().all()):
        return None  # its parts were made again differently
    if db.execute(select(StudyProgress.id).where(StudyProgress.scope_key == scope, StudyProgress.completed.is_(True),
                                                 StudyProgress.node_id == row.node_id)).first():
        return None  # studied: what comes next follows from progress
    return {"node_id": row.node_id, "title": row.title, "trail": trail, "budget_tokens": row.budget_tokens}
