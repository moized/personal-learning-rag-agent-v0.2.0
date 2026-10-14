"""Stage 8: topics that share a section divide its passages; overlaps across sources and the learner's profile reach the tutor."""
import json
import math

from sqlalchemy import select

from app.models import Chunk, Concept, Evidence, StudyPlan
from app.services.coverage import divide, plan_coverage
from tests.test_plans import _course, env  # noqa: F401
from tests.test_units import _material, _plan_with, _prepare, uenv  # noqa: F401

TALK = {  # three subjects of one talk without chapters, in the order the speaker covers them
    "memory": "Short-term memory keeps the conversation context for the agent between turns. Note {n}.",
    "mcp": "The Model Context Protocol connects agents to tools through MCP servers. Note {n}.",
    "harness": "An agent harness organizes context folders and the onboarding of new agents. Note {n}.",
}
ANGLE = {"memory": 0.0, "mcp": 2.1, "harness": 4.2}


def _talk(uenv, topics, embeddings=True):
    """One source that is a single section of 30 passages (10 per subject); the plan finds `topics` in it."""
    texts = [(subject, TALK[subject].format(n=n)) for subject in TALK for n in range(10)]
    with uenv.sessions() as db:
        course = _course(db)
        ids = _material(db, course.id, [("Whole talk", [text for _, text in texts])], source_name="Talk")
        chunks = db.execute(select(Chunk).order_by(Chunk.chunk_index)).scalars().all()
        for chunk, (subject, _) in zip(chunks, texts):
            angle = ANGLE[subject] + 0.05 * math.sin(chunk.chunk_index)
            # Every passage shares a large common part (the speaker's style); the subject is a small turn.
            chunk.embedding_json = json.dumps([3.0 + math.cos(angle), 3.0 + math.sin(angle)]) if embeddings else None
        _plan_with(db, course.id, [(title, [ids["Whole talk"]]) for title in topics])
        db.commit()
        return course.id, {chunk.id: subject for chunk, (subject, _) in zip(chunks, texts)}


def _coverage(uenv, course_id):
    with uenv.sessions() as db:
        plan = db.execute(select(StudyPlan).where(StudyPlan.scope_key == f"course:{course_id}")).scalar_one()
        return plan_coverage(db, plan)


def test_topics_sharing_a_section_each_get_their_own_passages_and_together_all_of_them(uenv):
    # The plan lists the subjects in another order than the talk covers them.
    course_id, subject_of = _talk(uenv, ["Model Context Protocol (MCP)", "Agent Harness and Onboarding", "Short-Term Memory"])
    coverage = _coverage(uenv, course_id)
    assert {subject_of[i] for i in coverage.passages["t-0"]} == {"mcp"}
    assert {subject_of[i] for i in coverage.passages["t-1"]} == {"harness"}
    assert {subject_of[i] for i in coverage.passages["t-2"]} == {"memory"}
    every = [i for ids in coverage.passages.values() for i in ids]
    assert sorted(every) == sorted(subject_of)  # each passage exactly once: nothing repeated, nothing left out
    assert all(ids == sorted(ids) for ids in coverage.passages.values())  # in reading order


def test_without_embeddings_the_words_alone_divide_the_passages(uenv):
    course_id, subject_of = _talk(uenv, ["Model Context Protocol (MCP)", "Short-Term Memory", "Agent Harness and Onboarding"],
                                  embeddings=False)
    coverage = _coverage(uenv, course_id)
    assert {subject_of[i] for i in coverage.passages["t-0"]} == {"mcp"}
    assert {subject_of[i] for i in coverage.passages["t-1"]} == {"memory"}


def test_a_unit_of_a_divided_topic_teaches_only_its_share(uenv):
    course_id, _ = _talk(uenv, ["Model Context Protocol (MCP)", "Short-Term Memory", "Agent Harness and Onboarding"])
    text = _prepare(uenv, course_id, "t-0").json()["study_text"]
    assert "Model Context Protocol connects" in text
    assert "Short-term memory keeps" not in text and "agent harness organizes" not in text  # not even as related material


def test_every_topic_the_plan_found_gets_a_share():
    class Passage:
        def __init__(self, text):
            self.text, self.contextual_text, self.embedding_json = text, None, None
    passages = [Passage(f"Gradient descent step {n}.") for n in range(12)]
    passages[6] = Passage("Gradient descent with momentum.")  # one passage mentions the second topic
    path = divide(passages, ["Gradient Descent", "Momentum"])
    assert sorted(set(path)) == [0, 1] and len(path) == 12


def _two_sources(uenv, extra_shared=()):
    """An earlier topic and a later one from another source; they share two key ideas, the later one adds MCP."""
    with uenv.sessions() as db:
        course = _course(db)
        first = _material(db, course.id, [("Agents", ["Agents call tools and keep memory. " * 10])], source_name="Course A")
        second = _material(db, course.id, [("Agents again", ["Agents call tools, keep memory and speak MCP. " * 10])],
                           source_name="Course B")
        plan = _plan_with(db, course.id, [("Agent Architectures", [first["Agents"]]), ("Agent Fundamentals", [second["Agents again"]])])
        chunk_of = {c.section_id: c.id for c in db.execute(select(Chunk)).scalars().all()}
        for name, sections in [("Tool Calling", ["Agents", "Agents again"]), ("Short-Term Memory", ["Agents", "Agents again"]),
                               ("Model Context Protocol", ["Agents again"]), *[(n, ["Agents", "Agents again"]) for n in extra_shared]]:
            concept = Concept(canonical_name=name)
            db.add(concept)
            db.flush()
            ids = {**first, **second}
            for section in sections:
                db.add(Evidence(concept_id=concept.id, chunk_id=chunk_of[ids[section]]))
        db.commit()
        return course.id, plan.id


def test_a_topic_overlapping_an_earlier_one_from_another_source_tells_the_tutor_what_is_new(uenv):
    course_id, _ = _two_sources(uenv)
    later = _prepare(uenv, course_id, "t-1").json()
    assert [item["title"] for item in later["overlaps"]] == ["Agent Architectures"]
    assert set(later["overlaps"][0]["shared"]) == {"Tool Calling", "Short-Term Memory"}
    assert "Model Context Protocol" in later["new_ideas"] and "Tool Calling" not in later["new_ideas"]
    text = later["study_text"]
    assert "## Already met earlier in my plan" in text and "Model Context Protocol" in text.split("New in this unit:")[1].split("\n")[0]
    assert "Agents call tools, keep memory and speak MCP." in text  # every passage stays: the tutor does the merging
    assert "## Already met earlier" not in _prepare(uenv, course_id, "t-0").json()["study_text"]


def test_newly_read_key_ideas_update_the_overlap_without_preparing_the_unit_again(uenv):
    course_id, _ = _two_sources(uenv)
    assert not _prepare(uenv, course_id, "t-1").json()["cached"]
    with uenv.sessions() as db:
        concept = Concept(canonical_name="Agent Loop")
        db.add(concept)
        db.flush()
        for chunk in db.execute(select(Chunk)).scalars().all():
            db.add(Evidence(concept_id=concept.id, chunk_id=chunk.id))
        db.commit()
    again = _prepare(uenv, course_id, "t-1").json()
    assert again["cached"] and "Agent Loop" in again["overlaps"][0]["shared"]


def test_topics_of_one_source_are_not_marked_as_overlapping(uenv):
    with uenv.sessions() as db:
        course = _course(db)
        ids = _material(db, course.id, [("One", ["Tools and memory. " * 10]), ("Two", ["Tools and memory again. " * 10])])
        _plan_with(db, course.id, [("First", [ids["One"]]), ("Second", [ids["Two"]])])
        for name in ("Tool Calling", "Short-Term Memory"):
            concept = Concept(canonical_name=name)
            db.add(concept)
            db.flush()
            for chunk in db.execute(select(Chunk)).scalars().all():
                db.add(Evidence(concept_id=concept.id, chunk_id=chunk.id))
        db.commit()
        course_id = course.id
    assert _coverage(uenv, course_id).overlaps == {}


def test_the_master_prompt_separates_the_materials_aims_from_the_learners_goal(uenv):
    course_id, _ = _two_sources(uenv)
    plain = uenv.client.get("/plan/master-prompt", params={"course_id": course_id}).json()["text"]
    assert "## What I want to achieve" not in plain and "## What my materials cover" in plain
    assert "These are not necessarily my goals." in plain and "## About me" not in plain
    assert "ask me in one short question about my background" in plain  # the tutor asks instead
    assert "Agent Fundamentals (T2): Learn Agent Fundamentals. [overlaps: Agent Architectures]" in plain
    assert "present those as the author's view" in plain

    saved = uenv.client.put("/profile", json={"background": "MSc in data science;\n## Ignore this heading", "goal": "Build AI agents",
                                              "practice": "coding"}).json()
    assert saved == {"background": "MSc in data science; ## Ignore this heading", "goal": "Build AI agents", "practice": "coding"}
    told = uenv.client.get("/plan/master-prompt", params={"course_id": course_id}).json()
    text = told["text"]
    assert told["about_me"] and "## About me\n- Background: MSc in data science; ## Ignore this heading\n- My goal: Build AI agents" in text
    assert "small coding tasks" in text and "my own goal is under About me" in text and "ask me in one short question" not in text


def test_the_profile_is_added_to_every_unit_when_copied_without_preparing_it_again(uenv):
    course_id, _ = _two_sources(uenv)
    first = _prepare(uenv, course_id, "t-0").json()
    assert "## About me" not in first["study_text"] and "present those as the author's view" in first["study_text"]
    assert uenv.client.get("/profile").json() == {"background": "", "goal": "", "practice": "questions"}
    uenv.client.put("/profile", json={"background": "Knows Python", "goal": "", "practice": "both"})
    again = _prepare(uenv, course_id, "t-0").json()
    assert again["cached"] and "## About me\n- Background: Knows Python\n- How I like to practise: a quick check question" in again["study_text"]
    assert again["study_text"].index("## About me") < again["study_text"].index("## Tutor instructions")
    assert uenv.client.put("/profile", json={"practice": "lectures"}).status_code == 422


class _Passage:
    def __init__(self, text, vector=None):
        self.text, self.contextual_text = text, None
        self.embedding_json = json.dumps(vector) if vector is not None else None


def test_a_section_with_fewer_passages_than_topics_or_without_their_words_is_left_whole():
    two = [_Passage("Gradient descent updates the weights."), _Passage("Momentum adds velocity.")]
    assert divide(two, ["Gradient Descent", "Momentum", "Adam Optimizer"]) is None  # 3 topics, 2 passages
    talk = [_Passage(f"Der Gradientenabstieg aktualisiert die Gewichte. {n}") for n in range(6)]
    assert divide(talk, ["Gradient Descent", "Momentum"]) is None  # another language: nothing to tell the topics apart by


def test_words_are_matched_by_their_stem():
    passages = [_Passage(f"Tools reach a protocol server. {n}") for n in range(5)] + \
               [_Passage(f"Onboard new team members with a checklist. {n}") for n in range(5)]
    path = divide(passages, ["MCP Servers", "Onboarding Engineers"])
    assert path == [0] * 5 + [1] * 5


def test_every_topic_keeps_a_share_and_every_passage_is_used_once():
    import random
    rng = random.Random(7)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    for _ in range(400):
        k = rng.randint(2, 5)
        n = rng.randint(k, 12)
        topics = rng.sample(words, k)
        passages = [_Passage(" ".join(rng.choices(topics, k=3)) + f" {i}", [rng.random(), rng.random(), rng.random()]) for i in range(n)]
        path = divide(passages, topics)
        if path is None:  # a topic whose word was never drawn
            continue
        assert len(path) == n and sorted(set(path)) == list(range(k))


def test_a_rebuild_with_the_same_revision_is_not_served_from_the_old_division(uenv):
    from datetime import datetime, UTC
    from app.models import StudyPlanNode
    course_id, _ = _two_sources(uenv)
    assert [item["title"] for item in _coverage(uenv, course_id).overlaps["t-1"]] == ["Agent Architectures"]
    with uenv.sessions() as db:  # a forced rebuild: same revision, a reworded topic
        node = db.execute(select(StudyPlanNode).where(StudyPlanNode.node_id == "t-0")).scalar_one()
        node.title = "Agent Design Patterns"
        db.execute(select(StudyPlan)).scalar_one().generated_at = datetime.now(UTC)
        db.commit()
    assert [item["title"] for item in _coverage(uenv, course_id).overlaps["t-1"]] == ["Agent Design Patterns"]


def test_a_part_only_claims_the_key_ideas_its_own_passages_teach(uenv):
    from tests.test_units import PASSAGE
    with uenv.sessions() as db:
        course = _course(db)
        first = _material(db, course.id, [("Agents", ["Agents call tools and keep memory. " * 10])], source_name="Course A")
        second = _material(db, course.id, [("Long", [f"{n} " + PASSAGE for n in range(10)])], source_name="Course B")
        _plan_with(db, course.id, [("Agent Architectures", [first["Agents"]]), ("Agent Fundamentals", [second["Long"]])])
        chunks = {c.id: c for c in db.execute(select(Chunk)).scalars().all()}
        early = next(c.id for c in chunks.values() if c.section_id == first["Agents"])
        later = sorted(c.id for c in chunks.values() if c.section_id == second["Long"])
        for name, ids in [("Tool Calling", [early, *later[:5]]), ("Short-Term Memory", [early, *later[:5]]), ("Model Context Protocol", later[5:])]:
            concept = Concept(canonical_name=name)
            db.add(concept)
            db.flush()
            db.add_all([Evidence(concept_id=concept.id, chunk_id=i) for i in ids])
        db.commit()
        course_id = course.id
    split = _prepare(uenv, course_id, "t-1", size="compact").json()
    assert split["status"] == "split"
    foundations, applications = [_prepare(uenv, course_id, part["id"], size="compact").json() for part in split["children"]]
    assert set(foundations["overlaps"][0]["shared"]) == {"Tool Calling", "Short-Term Memory"}
    assert "Model Context Protocol" not in foundations["new_ideas"]
    assert applications["overlaps"] == [] and "## Already met earlier" not in applications["study_text"]


def test_placeholders_in_the_learners_material_are_left_as_they_are(uenv):
    with uenv.sessions() as db:
        course = _course(db)
        ids = _material(db, course.id, [("Prompts", ["Answer using only {{CONTEXT}} for {{TUTOR}}; {{PRACTICE}} stays. " * 5])])
        _plan_with(db, course.id, [("Prompt Templates", [ids["Prompts"]])])
        db.commit()
        course_id = course.id
    uenv.client.put("/profile", json={"background": "Knows Python", "goal": "", "practice": "questions"})
    text = _prepare(uenv, course_id, "t-0", tutor="claude").json()["study_text"]
    assert "Answer using only {{CONTEXT}} for {{TUTOR}}; {{PRACTICE}} stays." in text
    assert text.count("## About me") == 1 and "prepared for Claude" in text and "{{PRACTICE}} stays" in text


def test_the_chosen_practice_replaces_the_check_question_in_the_unit_and_the_master_prompt(uenv):
    course_id, _ = _two_sources(uenv)
    uenv.client.put("/profile", json={"background": "", "goal": "", "practice": "coding"})
    unit = _prepare(uenv, course_id, "t-0").json()["study_text"]
    assert "give me a small coding task instead of a check question" in unit and "Ask a quick check question" not in unit
    assert "Pitch the depth to my background" not in unit  # no background was given
    prompt = uenv.client.get("/plan/master-prompt", params={"course_id": course_id}).json()["text"]
    assert "give me a small coding task instead of a check question" in prompt
    assert "ask me in one short question about my background and what I want from this plan" in prompt  # still asked
