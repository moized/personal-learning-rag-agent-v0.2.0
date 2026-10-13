"""What the learner tells their tutor about themselves, once: background, goal and how they like to practise.

It is added to every study unit when it is copied (not stored in prepared units, so changing it never
prepares anything again) and to the master prompt. Without it, the master prompt asks the tutor to ask.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from ..models import LearnerProfile

PRACTICE = {
    "questions": "a quick check question after each idea",
    "coding": "small coding tasks: after each idea, one to do instead of a check question, with my code reviewed",
    "both": "a quick check question after each idea, and a small coding task where the idea can be practised in code",
}
# The tutor's instruction for practice, in a unit's own tutor instructions and in the master prompt.
PRACTICE_INSTRUCTIONS = {
    "questions": "Ask a quick check question and wait for my answer before moving on.",
    "coding": "After each idea, give me a small coding task instead of a check question, and wait for my code and review it"
              " before moving on.",
    "both": "After each idea, ask a quick check question, add a small coding task where the idea can be practised in code, and"
            " wait for my answer before moving on.",
}
BACKGROUND_CHARS = 600
GOAL_CHARS = 300


def _line(value: str | None, limit: int) -> str:
    """One line, so the learner's text cannot add headings or list items of its own to a prompt."""
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def get_profile(db: Session) -> LearnerProfile | None:
    return db.get(LearnerProfile, 1)


def profile_json(profile: LearnerProfile | None) -> dict:
    return {"background": profile.background or "" if profile else "", "goal": profile.goal or "" if profile else "",
            "practice": profile.practice if profile and profile.practice in PRACTICE else "questions"}


def save_profile(db: Session, background: str, goal: str, practice: str) -> LearnerProfile:
    profile = get_profile(db) or LearnerProfile(id=1)
    profile.background = _line(background, BACKGROUND_CHARS) or None
    profile.goal = _line(goal, GOAL_CHARS) or None
    profile.practice = practice if practice in PRACTICE else "questions"
    db.add(profile)
    db.flush()
    return profile


def is_set(profile: LearnerProfile | None) -> bool:
    return bool(profile and (profile.background or profile.goal or (profile.practice and profile.practice != "questions")))


def practice_instruction(profile: LearnerProfile | None) -> str:
    return PRACTICE_INSTRUCTIONS.get(profile.practice if profile else None, PRACTICE_INSTRUCTIONS["questions"])


def missing(profile: LearnerProfile | None) -> list[str]:
    """What the tutor should ask about, because the learner has not said it."""
    return [what for what, known in (("my background", profile and profile.background), ("what I want from this plan", profile and profile.goal))
            if not known]


def about_me_lines(profile: LearnerProfile | None) -> list[str]:
    if not is_set(profile):
        return []
    lines = []
    if profile.background:
        lines.append(f"- Background: {_line(profile.background, BACKGROUND_CHARS)}")
    if profile.goal:
        lines.append(f"- My goal: {_line(profile.goal, GOAL_CHARS)}")
    lines.append(f"- How I like to practise: {PRACTICE.get(profile.practice or 'questions', PRACTICE['questions'])}")
    return lines


def about_me_block(profile: LearnerProfile | None) -> str:
    """The "About me" section of a study unit, or nothing."""
    lines = about_me_lines(profile)
    if not lines:
        return ""
    advice = (["Pitch the depth to my background: move quickly over what I likely know and slow down on what is new to me."]
              if profile.background else []) + (["Connect the material to my goal where it fits."] if profile.goal else [])
    return "## About me\n" + "\n".join(lines) + (f"\n{' '.join(advice)}" if advice else "") + "\n\n"
