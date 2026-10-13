"""The learner's own AI arranges the plan's topics into study sessions; LearnOS makes no AI call for it."""
from sqlalchemy import select

from app.models import PlanArrangement, StudyPlanNode, StudyProgress
from app.services.arrangement import group_id
from tests.test_plans import _course, env  # noqa: F401
from tests.test_units import _material, _plan_with, _prepare, uenv  # noqa: F401

LOOP, BRAND, AGAIN = "Agents run a loop of observe and act.", "Build a personal brand website.", "The agent loop repeats observe, think, act."


def _three(uenv):
    """Three topics from three sources; the first and the third teach the same thing."""
    with uenv.sessions() as db:
        course = _course(db)
        a = _material(db, course.id, [("Loops", [f"{LOOP} " * 10])], source_name="Course A")
        b = _material(db, course.id, [("Brand", [f"{BRAND} " * 10])], source_name="Course B")
        c = _material(db, course.id, [("Loops again", [f"{AGAIN} " * 10])], source_name="Course C")
        _plan_with(db, course.id, [("Agent Loop", [a["Loops"]]), ("Personal Brand", [b["Brand"]]), ("Agent Loop Again", [c["Loops again"]])])
        db.commit()
        return course.id


def _request(uenv, course_id):
    return uenv.client.get("/plan/organize-request", params={"course_id": course_id}).json()


def _reply(code, sessions):
    import json
    return "Here is my suggestion:\n```json\n" + json.dumps({"plan": code, "sessions": sessions}) + "\n```\nGood luck!"


ARRANGED = [{"topics": ["T1", "T3"], "title": "The agent loop, two views", "why": "Both teach the loop.", "optional": False},
            {"topics": ["T2"], "title": "Brand", "why": "Little for your goal.", "optional": "true"}]


def _arrange(uenv, course_id, sessions=ARRANGED):
    code = _request(uenv, course_id)["code"]
    return uenv.client.post("/plan/arrangement", json={"course_id": course_id, "reply": _reply(code, sessions)})


def test_the_request_lists_every_topic_with_a_short_id_and_costs_no_ai_call(uenv):
    course_id = _three(uenv)
    request = _request(uenv, course_id)
    text = request["text"]
    assert request["topics"] == 3 and f'"plan": "{request["code"]}"' in text
    assert "- T1 | Agent Loop | source: Course A" in text and "- T3 | Agent Loop Again | source: Course C" in text
    assert "## About me\n- Not stated." in text and uenv.state.ai.tasks == [] and uenv.state.ai.embeds == 0


def test_an_answer_arranges_sessions_with_the_optional_ones_last(uenv):
    course_id = _three(uenv)
    summary = _arrange(uenv, course_id, [ARRANGED[1], ARRANGED[0]]).json()
    assert summary == {"sessions": 2, "together": 1, "optional": 1, "left_out": [], "ignored": []}
    sessions = uenv.client.get("/syllabus", params={"course_id": course_id}).json()["sessions"]
    assert [s["topics"] for s in sessions] == [["t-0", "t-2"], ["t-1"]] and sessions[1]["optional"]
    assert sessions[0]["id"] == group_id(["t-0", "t-2"])


def test_a_combined_session_is_one_unit_with_every_passage_of_its_topics(uenv):
    course_id = _three(uenv)
    _arrange(uenv, course_id)
    unit = _prepare(uenv, course_id, group_id(["t-0", "t-2"])).json()
    text = unit["study_text"]
    own = text.split("## Source excerpts")[1].split("## Related material")[0]  # related material may come from anywhere
    assert unit["title"] == "The agent loop, two views" and LOOP in own and AGAIN in own and BRAND not in own
    assert "- Studies together: Agent Loop (from Course A); Agent Loop Again (from Course C)" in text
    assert "point out where the sources agree, differ or add something" in text
    assert unit["next"] == {"id": "t-1", "title": "Personal Brand", "depth": 0}
    # A topic of the session opened on its own continues after the session.
    assert _prepare(uenv, course_id, "t-0").json()["next"] == {"id": "t-1", "title": "Personal Brand", "depth": 0}


def test_answers_that_cannot_be_used_say_what_to_do(uenv):
    course_id = _three(uenv)
    stale = uenv.client.post("/plan/arrangement", json={"course_id": course_id, "reply": _reply("00000000", ARRANGED)})
    assert stale.status_code == 422 and "earlier version of your study plan" in stale.json()["detail"]
    junk = uenv.client.post("/plan/arrangement", json={"course_id": course_id, "reply": "I think you should study the loop first."})
    assert junk.status_code == 422 and "could not find your AI's arrangement" in junk.json()["detail"]
    nothing = _arrange(uenv, course_id, [{"topics": ["T9"], "title": "x"}])
    assert nothing.status_code == 422
    partial = _arrange(uenv, course_id, [{"topics": ["T3", "T3", "T7"], "title": "Again"}]).json()
    assert partial["ignored"] == ["T3", "T7"] and partial["left_out"] == ["Agent Loop", "Personal Brand"]
    sessions = uenv.client.get("/syllabus", params={"course_id": course_id}).json()["sessions"]
    assert [s["topics"] for s in sessions] == [["t-2"], ["t-0"], ["t-1"]]  # left-out topics keep a session of their own


def test_progress_of_a_session_and_its_topics_stays_in_step(uenv):
    course_id = _three(uenv)
    _arrange(uenv, course_id)
    session = group_id(["t-0", "t-2"])

    def tick(topic_id, title, completed=True):
        uenv.client.post("/study-progress", json={"topic": title, "topic_id": topic_id, "course_id": course_id, "completed": completed})
        return set(uenv.client.get("/study-progress", params={"course_id": course_id}).json()["completed_topic_ids"])

    assert tick(session, "The agent loop, two views") == {session, "t-0", "t-2"}
    assert tick("t-2", "Agent Loop Again", False) == {"t-0"}  # a topic un-ticked: the session is no longer studied
    assert tick("t-2", "Agent Loop Again") == {session, "t-0", "t-2"}  # studied one by one: the session is studied too
    assert tick(session, "The agent loop, two views", False) == set()


def test_the_master_prompt_follows_the_sessions_and_where_the_learner_left_off(uenv):
    course_id = _three(uenv)
    _arrange(uenv, course_id)
    session = group_id(["t-0", "t-2"])
    prompt = uenv.client.get("/plan/master-prompt", params={"course_id": course_id}).json()
    text = prompt["text"]
    assert prompt["next_topic"]["id"] == session and "The first topic is: The agent loop, two views." in text
    assert "## My study sessions, in order\n1. The agent loop, two views: Agent Loop + Agent Loop Again\n2. Personal Brand [optional]" in text
    assert "Agent Loop: Learn Agent Loop. [studied together: The agent loop, two views]" in text and "[optional]" in text
    _prepare(uenv, course_id, session)
    progress = uenv.client.get("/study-progress", params={"course_id": course_id}).json()
    assert progress["position"]["trail"] == [session]
    assert "I am partway through: The agent loop, two views." in uenv.client.get("/plan/master-prompt", params={"course_id": course_id}).json()["text"]


def test_an_arrangement_survives_a_rebuild_and_can_be_undone(uenv):
    course_id = _three(uenv)
    _arrange(uenv, course_id)
    with uenv.sessions() as db:  # the plan was rebuilt without the third topic
        db.delete(db.execute(select(StudyPlanNode).where(StudyPlanNode.node_id == "t-2")).scalar_one())
        db.commit()
    sessions = uenv.client.get("/syllabus", params={"course_id": course_id}).json()["sessions"]
    assert [s["id"] for s in sessions] == ["t-0", "t-1"]  # the session lost a topic: the other one stays, alone
    assert uenv.client.delete("/plan/arrangement", params={"course_id": course_id}).json() == {"deleted": True}
    assert uenv.client.get("/syllabus", params={"course_id": course_id}).json()["sessions"] is None
    with uenv.sessions() as db:
        assert db.query(PlanArrangement).count() == 0 and db.query(StudyProgress).count() == 0


def _tick(uenv, course_id, topic_id, title, completed=True):
    uenv.client.post("/study-progress", json={"topic": title, "topic_id": topic_id, "course_id": course_id, "completed": completed})
    return set(uenv.client.get("/study-progress", params={"course_id": course_id}).json()["completed_topic_ids"])


def test_un_marking_a_session_keeps_a_topic_studied_on_its_own(uenv):
    course_id = _three(uenv)
    _tick(uenv, course_id, "t-0", "Agent Loop")  # studied before the arrangement
    _arrange(uenv, course_id)
    session = group_id(["t-0", "t-2"])
    assert _tick(uenv, course_id, session, "The agent loop, two views") == {session, "t-0", "t-2"}
    assert _tick(uenv, course_id, session, "The agent loop, two views", False) == {"t-0"}


def test_applying_an_arrangement_brings_old_session_marks_in_line(uenv):
    course_id = _three(uenv)
    _arrange(uenv, course_id)
    session = group_id(["t-0", "t-2"])
    _tick(uenv, course_id, session, "The agent loop, two views")
    uenv.client.delete("/plan/arrangement", params={"course_id": course_id})
    _tick(uenv, course_id, "t-0", "Agent Loop", False)  # no arrangement now: the session's old mark stays behind
    _arrange(uenv, course_id)
    assert session not in set(uenv.client.get("/study-progress", params={"course_id": course_id}).json()["completed_topic_ids"])
    prompt = uenv.client.get("/plan/master-prompt", params={"course_id": course_id}).json()
    assert prompt["next_topic"]["id"] == session and "[done]" not in prompt["text"].split("## My study sessions")[1].split("##")[0]


def test_pasting_the_whole_chat_finds_the_ais_answer_after_the_request(uenv):
    course_id = _three(uenv)
    request = _request(uenv, course_id)
    chat = request["text"] + "\n\nChatGPT said:\n" + _reply(request["code"], ARRANGED)
    assert uenv.client.post("/plan/arrangement", json={"course_id": course_id, "reply": chat}).json()["together"] == 1


def test_a_reply_that_is_not_json_is_refused_quickly():
    import time
    import pytest
    from app.services.arrangement import ArrangementError, _json_in
    started = time.perf_counter()
    with pytest.raises(ArrangementError):
        _json_in("```{" * 50000)
    assert time.perf_counter() - started < 5


def test_a_combined_session_lists_every_idea_it_teaches_for_the_first_time(uenv):
    from app.models import Chunk, Concept, Evidence
    with uenv.sessions() as db:
        course = _course(db)
        sections = {}
        for source, title, text in [("A", "Topic A", "Alpha text. " * 10), ("B", "Topic B", "Bravo text. " * 10), ("C", "Topic C", "Charlie text. " * 10)]:
            sections[title] = _material(db, course.id, [(title, [text])], source_name=source)[title]
        _plan_with(db, course.id, [(title, [sections[title]]) for title in ("Topic A", "Topic B", "Topic C")])
        chunk = {c.section_id: c.id for c in db.execute(select(Chunk)).scalars().all()}
        for idea, titles in [("P", ["Topic A", "Topic C"]), ("Q", ["Topic A", "Topic C"]), ("R", ["Topic B", "Topic C"]), ("S", ["Topic B"]), ("T", ["Topic C"])]:
            concept = Concept(canonical_name=idea)
            db.add(concept)
            db.flush()
            db.add_all([Evidence(concept_id=concept.id, chunk_id=chunk[sections[t]]) for t in titles])
        db.commit()
        course_id = course.id
    _arrange(uenv, course_id, [{"topics": ["T1"], "title": "A"}, {"topics": ["T2", "T3"], "title": "B and C"}])
    unit = _prepare(uenv, course_id, group_id(["t-1", "t-2"])).json()
    assert [item["title"] for item in unit["overlaps"]] == ["Topic A"]
    assert {"R", "S", "T"} <= set(unit["new_ideas"]) and not {"P", "Q"} & set(unit["new_ideas"])


LATER_FIRST = [{"topics": ["T2", "T3"], "title": "Brand and loop", "why": "First.", "optional": False},
               {"topics": ["T1"], "title": "Loop", "why": "Then.", "optional": False}]


def test_an_applied_arrangement_starts_from_its_first_session_not_from_a_topic_opened_before(uenv):
    course_id = _three(uenv)
    _prepare(uenv, course_id, "t-0")  # the learner opened the plan's first topic, then arranged the plan
    _arrange(uenv, course_id, LATER_FIRST)
    first = group_id(["t-1", "t-2"])
    prompt = uenv.client.get("/plan/master-prompt", params={"course_id": course_id}).json()
    assert prompt["next_topic"]["id"] == first and "I have not started yet. The first topic is: Brand and loop." in prompt["text"]
    assert uenv.client.get("/study-progress", params={"course_id": course_id}).json()["position"] is None
    unit = _prepare(uenv, course_id, first).json()  # session 1: every topic assigned to it, in one unit
    own = unit["study_text"].split("## Source excerpts")[1].split("## Related material")[0]
    assert BRAND in own and AGAIN in own and LOOP not in own and unit["next"]["id"] == "t-0"
    assert uenv.client.get("/study-progress", params={"course_id": course_id}).json()["position"]["trail"] == [first]
    # Keeping progress in step updates the arrangement row; a session opened after applying still counts.
    _tick(uenv, course_id, "t-0", "Agent Loop")
    assert uenv.client.get("/plan/master-prompt", params={"course_id": course_id}).json()["next_topic"]["id"] == first


def test_reopening_the_same_topic_after_arranging_counts_as_where_the_learner_is(uenv):
    course_id = _three(uenv)
    _prepare(uenv, course_id, "t-0")
    _arrange(uenv, course_id, LATER_FIRST)
    _prepare(uenv, course_id, "t-0")  # the learner chose to continue it anyway
    assert uenv.client.get("/plan/master-prompt", params={"course_id": course_id}).json()["next_topic"]["id"] == "t-0"


def test_topics_the_plan_gains_after_arranging_are_flagged_so_the_learner_can_ask_again(uenv):
    course_id = _three(uenv)
    _arrange(uenv, course_id, [{"topics": ["T1", "T3"], "title": "Loops"}])
    sessions = uenv.client.get("/syllabus", params={"course_id": course_id}).json()["sessions"]
    assert [(s["topics"], s["added"]) for s in sessions] == [(["t-0", "t-2"], False), (["t-1"], True)]
    assert _request(uenv, course_id)["topics"] == 3  # a new request lists every topic, so the AI can place the new one
