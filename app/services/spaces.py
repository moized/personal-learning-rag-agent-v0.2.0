"""Study spaces. A source is filed in one space, its primary place, and may also be shown in others."""
from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..models import Source, SourceSpace


def in_space(course_id: int):
    """SQL condition on Source: filed in the space, or also shown there."""
    return or_(Source.course_id == course_id,
               Source.id.in_(select(SourceSpace.source_id).where(SourceSpace.course_id == course_id)))


def memberships(db: Session) -> dict[int, set[int]]:
    """Every source's spaces: where it is filed plus where it is also shown."""
    spaces: dict[int, set[int]] = {}
    for source_id, course_id in db.execute(select(Source.id, Source.course_id)).all():
        spaces[source_id] = {course_id} if course_id is not None else set()
    for source_id, course_id in db.execute(select(SourceSpace.source_id, SourceSpace.course_id)).all():
        spaces.setdefault(source_id, set()).add(course_id)
    return spaces


def spaces_of(db: Session, source_ids) -> set[int]:
    """Every space these sources are filed in or shown in: the plans that depend on them."""
    source_ids = list(source_ids)
    if not source_ids:
        return set()
    primary = {c for c in db.execute(select(Source.course_id).where(Source.id.in_(source_ids))).scalars().all() if c is not None}
    return primary | set(db.execute(select(SourceSpace.course_id).where(SourceSpace.source_id.in_(source_ids))).scalars().all())


def also_in(db: Session, source_ids: list[int]) -> dict[int, list[int]]:
    """The spaces each source is also shown in, besides the one it is filed in."""
    if not source_ids:
        return {}
    shown: dict[int, list[int]] = {}
    for source_id, course_id in db.execute(select(SourceSpace.source_id, SourceSpace.course_id)
                                           .where(SourceSpace.source_id.in_(source_ids)).order_by(SourceSpace.id)).all():
        shown.setdefault(source_id, []).append(course_id)
    return shown
