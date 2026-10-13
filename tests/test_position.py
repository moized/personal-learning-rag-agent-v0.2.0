"""Where the learner left off: the session opened last is where they continue, even with nothing marked studied."""
from sqlalchemy import select

from app.models import StudyPlanNode, StudyPosition
from app.services.cleanup import forget_scope
from tests.test_plans import _course, env  # noqa: F401
from tests.test_units import _big, _prepare, _setup, uenv  # noqa: F401


def _progress(uenv, course_id):
    return uenv.client.get("/study-progress", params={"course_id": course_id}).json()


def _prompt(uenv, course_id):
    return uenv.client.get("/plan/master-prompt", params={"course_id": course_id}).json()


def test_an_opened_part_is_where_the_learner_continues_after_a_restart(uenv):
    course_id = _big(uenv)
    split = _prepare(uenv, course_id, "t-0", size="compact").json()
    assert split["status"] == "split" and _progress(uenv, course_id)["position"] is None  # a list of parts is not a session
    second = split["children"][1]
    assert _prepare(uenv, course_id, second["id"], size="compact").json()["status"] == "unit"
    position = _progress(uenv, course_id)["position"]
    assert position == {"node_id": second["id"], "title": second["title"], "trail": ["t-0", second["id"]], "budget_tokens": 4000}
    prompt = _prompt(uenv, course_id)
    assert prompt["started"] and prompt["next_topic"]["id"] == "t-0"
    assert "I am partway through: Gradient Descent" in prompt["text"]


def test_a_later_topic_opened_without_finishing_earlier_ones_is_where_the_learner_is(uenv):
    course_id = _setup(uenv, [("Loss", ["Loss measures error. " * 20]), ("Momentum", ["Momentum adds velocity. " * 20])],
                       [("Loss Functions", ["Loss"]), ("Momentum", ["Momentum"])])
    assert _prompt(uenv, course_id)["next_topic"]["id"] == "t-0"
    _prepare(uenv, course_id, "t-1")
    assert _prompt(uenv, course_id)["next_topic"]["id"] == "t-1"


def test_once_the_opened_session_is_studied_progress_decides_what_is_next(uenv):
    course_id = _setup(uenv, [("Loss", ["Loss measures error. " * 20]), ("Momentum", ["Momentum adds velocity. " * 20])],
                       [("Loss Functions", ["Loss"]), ("Momentum", ["Momentum"])])
    _prepare(uenv, course_id, "t-0")
    uenv.client.post("/study-progress", json={"topic": "Loss Functions", "course_id": course_id, "completed": True, "topic_id": "t-0"})
    assert _progress(uenv, course_id)["position"] is None
    assert _prompt(uenv, course_id)["next_topic"]["id"] == "t-1"


def test_a_position_outside_the_current_plan_is_ignored_and_forgotten_with_its_plan(uenv):
    course_id = _setup(uenv, [("Loss", ["Loss measures error. " * 20])], [("Loss Functions", ["Loss"])])
    _prepare(uenv, course_id, "t-0")
    with uenv.sessions() as db:
        db.execute(select(StudyPlanNode).where(StudyPlanNode.node_id == "t-0")).scalar_one().node_id = "t-9"
        db.commit()
    assert _progress(uenv, course_id)["position"] is None
    with uenv.sessions() as db:
        forget_scope(db, f"course:{course_id}")
        db.commit()
        assert db.query(StudyPosition).count() == 0
