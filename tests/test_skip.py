"""Skipping topics, and the tutor's answer about where to start and what to skip."""
import json

from app.services.arrangement import group_id
from tests.test_arrangement import AGAIN, ARRANGED, LOOP, _arrange, _three  # noqa: F401
from tests.test_plans import _course, env  # noqa: F401
from tests.test_units import _big, _prepare, uenv  # noqa: F401


def _skip(uenv, course_id, node_id, skipped=True):
    return uenv.client.post("/plan/skip", json={"course_id": course_id, "node_id": node_id, "skipped": skipped})


def _prompt(uenv, course_id):
    return uenv.client.get("/plan/master-prompt", params={"course_id": course_id}).json()


def _answer(code, start="", skip=()):
    return "Sure! Here it is:\n```json\n" + json.dumps({"plan": code, "start": start, "skip": list(skip)}) + "\n```"


def test_a_skipped_topic_is_never_next_and_not_counted_until_brought_back(uenv):
    course_id = _three(uenv)
    assert _skip(uenv, course_id, "t-0").json() == {"skipped_ids": ["t-0"]}
    prompt = _prompt(uenv, course_id)
    assert prompt["next_topic"]["id"] == "t-1" and prompt["topics"] == 2 and prompt["skipped"] == 1
    assert "Agent Loop (T1): Learn Agent Loop. [skipped]" in prompt["text"]
    assert "I chose to skip 1 topic." in prompt["text"] and "do not teach them unless I ask" in prompt["text"]
    assert _prepare(uenv, course_id, "t-1").json()["next"]["id"] == "t-2"
    assert uenv.client.get("/study-progress", params={"course_id": course_id}).json()["skipped_ids"] == ["t-0"]
    _skip(uenv, course_id, "t-0", False)
    back = _prompt(uenv, course_id)
    assert back["topics"] == 3 and back["skipped"] == 0 and "[skipped]" not in back["text"]
    assert back["next_topic"]["id"] == "t-1"  # the learner opened t-1 last: that is where they continue


def test_skipping_changes_what_comes_next_without_preparing_units_again(uenv):
    course_id = _big(uenv)  # t-0 splits into parts at the compact size
    split = _prepare(uenv, course_id, "t-0", size="compact").json()
    last = split["children"][-1]["id"]
    assert _prepare(uenv, course_id, last, size="compact").json()["next"]["id"] == "t-1"
    _skip(uenv, course_id, "t-1")
    again = _prepare(uenv, course_id, last, size="compact").json()
    assert again["cached"] and again["next"] is None and "- Comes next:" not in again["study_text"]


def test_a_session_can_be_skipped_and_a_skipped_topic_leaves_its_session(uenv):
    course_id = _three(uenv)
    _arrange(uenv, course_id)
    session = group_id(["t-0", "t-2"])
    _skip(uenv, course_id, "t-2")  # one topic of the session: the session keeps the other one
    own = _prepare(uenv, course_id, session).json()["study_text"].split("## Source excerpts")[1].split("## Related material")[0]
    assert LOOP in own and AGAIN not in own
    _skip(uenv, course_id, session)
    prompt = _prompt(uenv, course_id)
    assert prompt["next_topic"]["id"] == "t-1" and "[skipped]" in prompt["text"].split("## My study sessions")[1].split("##")[0]
    _skip(uenv, course_id, session, False)  # bringing the session back brings back its topics
    assert uenv.client.get("/study-progress", params={"course_id": course_id}).json()["skipped_ids"] == []


def test_the_tutor_chooses_where_to_start_and_what_to_skip(uenv):
    course_id = _three(uenv)
    prompt = _prompt(uenv, course_id)
    text, code = prompt["text"], prompt["code"]
    assert f'{{"plan": "{code}", "start": "T?", "skip": ["T?"]}}' in text
    assert "then suggest where I should start and which topics I could skip" in text
    result = uenv.client.post("/plan/start-and-skip", json={"course_id": course_id, "reply": text + "\n\nTutor:\n" + _answer(code, "T3", ["T1", "T9"])}).json()
    assert result == {"start": "Agent Loop Again", "skipped": ["Agent Loop"], "ignored": ["T9"]}
    after = _prompt(uenv, course_id)
    assert after["next_topic"]["id"] == "t-2" and not after["started"]
    assert "I have not started yet. I want to start with: Agent Loop Again." in after["text"]
    position = uenv.client.get("/study-progress", params={"course_id": course_id}).json()["position"]
    assert position["trail"] == ["t-2"] and position["budget_tokens"] == 0  # chosen, not opened yet


def test_starting_with_a_skipped_topic_brings_it_back_and_a_combined_session_starts_whole(uenv):
    course_id = _three(uenv)
    _arrange(uenv, course_id)
    _skip(uenv, course_id, "t-2")
    code = _prompt(uenv, course_id)["code"]
    uenv.client.post("/plan/start-and-skip", json={"course_id": course_id, "reply": _answer(code, "T3")})
    progress = uenv.client.get("/study-progress", params={"course_id": course_id}).json()
    assert progress["skipped_ids"] == [] and progress["position"]["trail"] == [group_id(["t-0", "t-2"])]


def test_answers_that_cannot_be_used_say_what_to_do(uenv):
    course_id = _three(uenv)
    code = _prompt(uenv, course_id)["code"]

    def post(reply):
        return uenv.client.post("/plan/start-and-skip", json={"course_id": course_id, "reply": reply})

    assert "earlier version" in post(_answer("00000000", "T1")).json()["detail"]
    assert "could not find your tutor's answer" in post(_answer(code, "T?", ["T?"])).json()["detail"]
    assert "does not name any of your topics" in post(_answer(code, "T7", ["T8"])).json()["detail"]
    assert uenv.client.post("/plan/skip", json={"course_id": course_id, "node_id": "t-9"}).status_code == 404
