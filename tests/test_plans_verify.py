"""Regression tests for the verification of the Stage 3 review fixes."""
import json

from app.providers.base import ProviderQuotaExceededError
from app.services.learning_paths import _normalize_nodes
from tests.plan_jobs import run_plan_jobs
from tests.test_plans import NODE, PlanAI, _add_source, _course, _plan, env  # noqa: F401
from tests.test_plans_review import _build, _change_content


def _node(local_id, title, sections, kind="study"):
    return {"id": local_id, "parent_id": "none", "type": kind, "title": title, "source_ids": [1], "section_ids": sections, **NODE}


def test_an_explicitly_reused_id_beats_an_earlier_section_match():
    previous = {("study", frozenset({(10, "Lecture 1")})): ("t-X", "Gradient Descent")}
    raw = [_node("new", "Optimization overview", [10]), _node("t-X", "Gradient Descent", [10, 11])]
    result = {n["title"]: n["node_id"] for n in _normalize_nodes(raw, "path-1", {1}, {1: "x"}, frozenset({"t-X"}), frozenset({10, 11}),
                                                                   None, previous, {10: "Lecture 1", 11: "Lecture 2"})}
    assert result["Gradient Descent"] == "t-X" and result["Optimization overview"] != "t-X"


def test_the_same_title_wins_when_several_topics_cover_the_previous_sections():
    previous = {("study", frozenset({(10, "Lecture 1")})): ("t-X", "Gradient Descent")}
    raw = [_node("1", "Loss Surfaces", [10]), _node("2", "Gradient Descent", [10])]
    result = {n["title"]: n["node_id"] for n in _normalize_nodes(raw, "path-1", {1}, {1: "x"}, frozenset(), frozenset({10}),
                                                                   None, previous, {10: "Lecture 1"})}
    assert result["Gradient Descent"] == "t-X" and result["Loss Surfaces"] != "t-X"


def test_a_recycled_section_id_with_another_title_does_not_inherit_a_topic():
    previous = {("study", frozenset({(2, "Momentum lecture")})): ("t-M", "Momentum")}
    raw = [_node("1", "Attention", [2])]
    result = _normalize_nodes(raw, "path-1", {1}, {1: "x"}, frozenset(), frozenset({2}), None, previous, {2: "Attention lecture"})
    assert result[0]["node_id"] != "t-M" and result[0]["section_titles"] == ["Attention lecture"]


class QuotaOn(PlanAI):
    def __init__(self, kind):
        super().__init__(relation="SAME_COURSE")
        self.kind = kind

    def generate_json(self, prompt, task="extraction", schema=None):
        if prompt.startswith(self.kind):
            raise ProviderQuotaExceededError("daily quota exhausted")
        return super().generate_json(prompt, task, schema)


def _two_source_plan(env):
    with env.sessions() as db:
        course = _course(db)
        a = _add_source(db, course.id, "A", ["Gradient Descent"])
        _add_source(db, course.id, "B", ["Momentum"])
        db.commit()
        return course.id, a.id


def test_quota_in_the_merge_keeps_the_previous_plan(env):
    course_id, a_id = _two_source_plan(env)
    good = _build(env, course_id)
    with env.sessions() as db:
        _change_content(db, a_id, "Learning Rate")
        db.commit()
    env.state.ai = QuotaOn("Synthesize")
    after = _build(env, course_id)
    assert after["status"] == "failed" and "quota" in after["error"]
    assert [n["id"] for n in after["nodes"]] == [n["id"] for n in good["nodes"]]


def test_quota_while_comparing_sources_keeps_the_previous_plan(env):
    course_id, _ = _two_source_plan(env)
    good = _build(env, course_id)
    with env.sessions() as db:
        _add_source(db, course_id, "C", ["Attention"])  # a new pair to compare
        db.commit()
    env.state.ai = QuotaOn("Classify")
    after = _build(env, course_id)
    assert after["status"] == "failed" and [n["id"] for n in after["nodes"]] == [n["id"] for n in good["nodes"]]


def test_the_merge_sees_source_namespaced_ids(env):
    course_id, _ = _two_source_plan(env)
    _build(env, course_id)
    merge = next(prompt for kind, prompt in env.state.ai.prompts if kind == "merge")
    inputs = json.loads(merge.split("relationship evidence: ")[1].split("\nSource identity classifications")[0])
    ids = [node["id"] for path in inputs for curriculum in path["source_curricula"] for node in curriculum["nodes"]]
    assert ids and all(node_id.startswith("s") and ":" in node_id for node_id in ids) and len(ids) == len(set(ids))
