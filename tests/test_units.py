"""Stage 4: plan topics prepared for one study session, split into parts when they are too large."""
import json
import re
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app import main
from app.models import Chunk, Concept, Document, Evidence, Section, Source, StudyPlan, StudyPlanNode, TopicDetail
from app.providers.base import ProviderQuotaExceededError
from app.services import units
from tests.test_plans import PlanAI, _add_source, _course, env  # noqa: F401
from tests.test_plans_review import _build

PASSAGE = "x" * 1497  # 500 tokens by the three-characters-per-token estimate


class UnitAI:
    """Embeds everything alike and names the parts of a split; counts what it is asked."""

    def __init__(self, split="ok"):
        self.split, self.tasks, self.embeds = split, [], 0

    def embed(self, texts):
        self.embeds += 1
        return [[1.0, 0.0] for _ in texts]

    def generate_json(self, prompt, task="extraction", schema=None):
        self.tasks.append(task)
        assert task == "outline_split"
        ids = [g["id"] for g in json.loads(re.search(r"size in tokens\): (\[.*\])\n", prompt).group(1))]
        if self.split == "quota":
            raise ProviderQuotaExceededError("daily quota exhausted")
        if self.split == "gap":  # skips a group: must not be accepted
            return {"parts": [{"title": "A", "purpose": "", "group_ids": ids[:1]}, {"title": "B", "purpose": "", "group_ids": ids[2:]}]}
        if self.split == "broken":  # fails the same way every time: not worth another call
            raise RuntimeError("All Gemini JSON models failed: fast: invalid JSON")
        if self.split == "each":  # one part per group
            return {"parts": [{"title": f"Piece {n}", "purpose": "", "group_ids": [g]} for n, g in enumerate(ids, 1)]}
        half = (len(ids) + 1) // 2
        return {"parts": [{"title": "Foundations", "purpose": "The basics.", "group_ids": ids[:half]},
                          {"title": "Applications", "purpose": "Using them.", "group_ids": ids[half:]}]}


@pytest.fixture
def uenv(env, monkeypatch):
    env.state.ai = UnitAI()
    monkeypatch.setattr(main, "provider", lambda: env.state.ai)
    return env


def _material(db, course_id, sections, parents=False, source_name="Book"):
    """One document whose passages are divided into titled sections; returns {title: section id}."""
    source = Source(name=source_name, source_type="pdf", status="done", course_id=course_id)
    db.add(source)
    db.flush()
    doc = Document(source_id=source.id, external_id="book", title="Book")
    db.add(doc)
    db.flush()
    ids, index, leaves = {}, 0, []
    for position, (title, texts) in enumerate(sections):
        section = Section(source_id=source.id, document_id=doc.id, position=position, title=title, kind="chapter", chunk_count=len(texts))
        db.add(section)
        db.flush()
        ids[title] = section.id
        concept = Concept(canonical_name=f"{title} concept")
        db.add(concept)
        db.flush()
        for text in texts:
            chunk = Chunk(document_id=doc.id, chunk_index=index, chunk_kind="leaf", text=text, section_id=section.id,
                          embedding_status="done", embedding_json=json.dumps([1.0, 0.0]))
            db.add(chunk)
            db.flush()
            db.add(Evidence(concept_id=concept.id, chunk_id=chunk.id))
            leaves.append(chunk)
            index += 1
    if parents:  # windows of two passages that cross section boundaries, as in a PDF
        for start in range(0, len(leaves) - 1):
            parent = Chunk(document_id=doc.id, chunk_index=1000 + start, chunk_kind="parent",
                           text=leaves[start].text + "\n" + leaves[start + 1].text)
            db.add(parent)
            db.flush()
            if start % 2 == 0:
                leaves[start].parent_chunk_id = parent.id
                leaves[start + 1].parent_chunk_id = parent.id
    db.flush()
    return ids


def _plan_with(db, course_id, topics, revision="r1"):
    plan = StudyPlan(scope_key=f"course:{course_id}", revision=revision, syllabus="")
    db.add(plan)
    db.flush()
    db.add(StudyPlanNode(plan_id=plan.id, node_id="path-1", parent_node_id=None, node_type="outline", path_id="path-1",
                         title="ML Engineer", metadata_json=json.dumps({"purpose": "Train and ship models"})))
    for number, (title, section_ids) in enumerate(topics):
        db.add(StudyPlanNode(plan_id=plan.id, node_id=f"t-{number}", parent_node_id="path-1", node_type="study", path_id="path-1",
                             title=title, metadata_json=json.dumps({"purpose": f"Learn {title}.", "section_ids": section_ids,
                                                                    "prerequisites": "Calculus" if number else ""})))
    db.flush()
    return plan


def _setup(uenv, sections, topics, parents=False, source_name="Book"):
    with uenv.sessions() as db:
        course = _course(db)
        ids = _material(db, course.id, sections, parents, source_name)
        _plan_with(db, course.id, [(title, [ids[name] for name in names]) for title, names in topics])
        db.commit()
        return course.id


def _prepare(uenv, course_id, node_id, size="standard", tutor="general"):
    return uenv.client.post("/plan/topics/prepare", json={"course_id": course_id, "node_id": node_id, "size": size, "tutor": tutor})


def _big(uenv, parents=False):
    """Topic t-0 has 10 passages of about 500 tokens: too large for a compact session (3,000 tokens for sources)."""
    return _setup(uenv, [("Optimization", [f"{n} " + PASSAGE for n in range(10)]), ("Regularization", ["Dropout drops units. " * 5])],
                  [("Gradient Descent", ["Optimization"]), ("Dropout", ["Regularization"])], parents)


def test_a_topic_that_fits_is_a_unit_made_without_an_ai_call_and_then_cached(uenv):
    course_id = _setup(uenv, [("Loss", ["Loss measures error. " * 20]), ("Momentum", ["Momentum adds velocity. " * 20])],
                       [("Loss Functions", ["Loss"]), ("Momentum", ["Momentum"])])
    first = _prepare(uenv, course_id, "t-0", tutor="claude").json()
    assert first["status"] == "unit" and not first["cached"] and first["children"] is None
    text = first["study_text"]
    assert "Loss measures error." in text and "Learning objective: ML Engineer: Train and ship models" in text
    assert "- This unit: ML Engineer > Loss Functions" in text and "- Comes next: Momentum" in text and "prepared for Claude" in text
    assert uenv.state.ai.tasks == [] and uenv.state.ai.embeds == 1  # only the related-material search
    again = _prepare(uenv, course_id, "t-0", tutor="gemini").json()
    assert again["cached"] and uenv.state.ai.embeds == 1 and "prepared for Gemini" in again["study_text"]


def test_related_material_is_marked_and_never_repeats_the_units_own_text(uenv):
    course_id = _setup(uenv, [("Loss", ["Loss measures error. " * 20, "Squared loss. " * 20]), ("Momentum", ["Momentum adds velocity. " * 20])],
                       [("Loss Functions", ["Loss"]), ("Momentum", ["Momentum"])], parents=True)
    result = _prepare(uenv, course_id, "t-0").json()
    own = [s for s in result["sources"] if not s["is_related"]]
    related = [s for s in result["sources"] if s["is_related"]]
    assert [s["section"] for s in own] == ["Loss", "Loss"]
    with uenv.sessions() as db:
        own_ids = {s["chunk_id"] for s in own}
        for ref in related:
            chunk = db.get(Chunk, ref["chunk_id"])
            covered = {chunk.id, chunk.parent_chunk_id} | set(db.execute(select(Chunk.id).where(Chunk.parent_chunk_id == chunk.id)).scalars())
            assert not covered & own_ids
            assert all(db.get(Chunk, i).parent_chunk_id not in covered for i in own_ids)
    assert related and "## Related material from elsewhere in this space" in result["study_text"]
    assert result["study_text"].count("Loss measures error.") == 20


def test_a_large_topic_is_split_into_named_parts_with_one_ai_call(uenv):
    course_id = _big(uenv)
    result = _prepare(uenv, course_id, "t-0", size="compact").json()
    assert result["status"] == "split" and "divided into 2 parts" in result["reason"]
    assert [c["title"] for c in result["children"]] == ["Foundations", "Applications"]
    assert uenv.state.ai.tasks == ["outline_split"] and uenv.state.ai.embeds == 0
    with uenv.sessions() as db:
        rows = db.execute(select(TopicDetail).where(TopicDetail.parent_node_id == "t-0").order_by(TopicDetail.id)).scalars().all()
        passages = [i for row in rows for i in json.loads(row.chunk_ids_json)]
        section = db.execute(select(Section.id).where(Section.title == "Optimization")).scalar_one()
        expected = db.execute(select(Chunk.id).where(Chunk.section_id == section).order_by(Chunk.chunk_index)).scalars().all()
    assert passages == expected  # every passage once, in reading order
    again = _prepare(uenv, course_id, "t-0", size="compact").json()
    assert again["cached"] and uenv.state.ai.tasks == ["outline_split"]
    assert _prepare(uenv, course_id, "t-0", size="xl").json()["status"] == "unit"  # sizes are prepared separately


def test_parts_open_as_units_that_know_their_place(uenv):
    course_id = _big(uenv)
    first, second = _prepare(uenv, course_id, "t-0", size="compact").json()["children"]
    part = _prepare(uenv, course_id, first["id"], size="compact").json()
    assert part["status"] == "unit" and part["depth"] == 1 and part["trail"] == ["ML Engineer", "Gradient Descent"]
    assert "This unit: ML Engineer > Gradient Descent > Foundations" in part["study_text"] and "Comes next: Applications" in part["study_text"]
    assert part["omitted_passages"] == 0 and part["estimated_tokens"] <= units.SESSION_SIZES["compact"]
    last = _prepare(uenv, course_id, second["id"], size="compact").json()
    assert "Comes next: Dropout" in last["study_text"]
    listed = _prepare(uenv, course_id, "t-0", size="compact").json()["children"]
    assert [c["status"] for c in listed] == ["unit", "unit"]
    assert uenv.client.post("/study-progress", json={"topic": first["title"], "topic_id": first["id"], "course_id": course_id}).status_code == 200
    assert first["id"] in uenv.client.get("/study-progress", params={"course_id": course_id}).json()["completed_topic_ids"]


def test_an_unavailable_ai_gives_a_plain_split_that_is_prepared_again_later(uenv):
    course_id = _big(uenv)
    uenv.state.ai = UnitAI(split="quota")
    result = _prepare(uenv, course_id, "t-0", size="compact").json()
    assert result["status"] == "split" and result["provisional"] and "Open it again later" in result["warning"]
    assert all(c["title"].startswith("Gradient Descent, part ") for c in result["children"])
    uenv.state.ai = UnitAI()
    retried = _prepare(uenv, course_id, "t-0", size="compact").json()
    assert not retried["cached"] and [c["title"] for c in retried["children"]] == ["Foundations", "Applications"]
    assert "provisional" not in retried


def test_an_invalid_ai_split_is_replaced_by_a_plain_split_that_is_kept(uenv):
    course_id = _big(uenv)
    uenv.state.ai = UnitAI(split="gap")
    result = _prepare(uenv, course_id, "t-0", size="compact").json()
    assert result["status"] == "split" and "provisional" not in result
    assert [c["title"].split(":")[0] for c in result["children"]] == ["Gradient Descent, part 1", "Gradient Descent, part 2"]
    assert _prepare(uenv, course_id, "t-0", size="compact").json()["cached"] and uenv.state.ai.tasks == ["outline_split"]


def test_at_the_depth_limit_a_topic_becomes_a_unit_with_what_fits(uenv, monkeypatch):
    monkeypatch.setattr(units, "MAX_SPLIT_DEPTH", 0)
    course_id = _big(uenv)
    result = _prepare(uenv, course_id, "t-0", size="compact").json()
    assert result["status"] == "unit" and result["omitted_passages"] > 0 and "do not fit" in result["note"]
    assert result["estimated_tokens"] <= units.SESSION_SIZES["compact"] and uenv.state.ai.tasks == []


def test_a_passage_larger_than_the_session_is_cut_not_dropped(uenv):
    course_id = _setup(uenv, [("Huge", ["y" * 30000])], [("Everything", ["Huge"])])
    result = _prepare(uenv, course_id, "t-0", size="compact").json()
    assert result["status"] == "unit" and result["truncated"] and "only its beginning" in result["note"]
    assert "yyyy" in result["study_text"] and result["estimated_tokens"] <= units.SESSION_SIZES["compact"]


def test_a_topic_with_little_material_is_marked_thin(uenv):
    course_id = _setup(uenv, [("Stub", ["Short."])], [("Stub Topic", ["Stub"])])
    result = _prepare(uenv, course_id, "t-0").json()
    assert result["status"] == "thin" and "very little" in result["note"]


def test_a_plan_change_prepares_topics_again_and_old_parts_ask_for_their_parent(uenv):
    course_id = _big(uenv)
    children = _prepare(uenv, course_id, "t-0", size="compact").json()["children"]
    with uenv.sessions() as db:
        db.execute(select(StudyPlan)).scalar_one().revision = "r2"
        db.commit()
    stale = _prepare(uenv, course_id, children[0]["id"], size="compact")
    assert stale.status_code == 409 and "parent topic" in stale.json()["detail"]
    again = _prepare(uenv, course_id, "t-0", size="compact").json()
    # Its material did not change, so the same parts still apply: no new AI call, same ids (and progress).
    assert uenv.state.ai.tasks == ["outline_split"] and [c["id"] for c in again["children"]] == [c["id"] for c in children]
    reopened = _prepare(uenv, course_id, children[0]["id"], size="compact").json()
    assert reopened["status"] == "unit" and not reopened["cached"]  # its text says where it fits, so it is made again


def test_outline_unknown_and_invalid_requests_are_refused(uenv):
    course_id = _setup(uenv, [("Loss", ["Loss measures error. " * 20])], [("Loss Functions", ["Loss"])])
    assert _prepare(uenv, course_id, "path-1").status_code == 422
    assert _prepare(uenv, course_id, "t-404").status_code == 404
    assert _prepare(uenv, course_id + 1, "t-0").status_code == 404
    assert _prepare(uenv, course_id, "t-0", size="huge").status_code == 422
    assert _prepare(uenv, course_id, "t-0", tutor="someone").status_code == 422


def test_a_topic_of_a_built_plan_is_prepared_from_its_sections(uenv):
    uenv.state.ai = PlanAI()
    with uenv.sessions() as db:
        course = _course(db)
        _add_source(db, course.id, "Lectures", ["Gradient Descent", "Backpropagation"])
        db.commit()
        course_id = course.id
    plan = _build(uenv, course_id)
    study = [n for n in plan["nodes"] if n["type"] == "study"]
    result = _prepare(uenv, course_id, study[0]["id"]).json()
    assert result["status"] in {"unit", "thin"} and f"{study[0]['title']} explained." in result["study_text"]
    assert [s["section"] for s in result["sources"] if not s["is_related"]] == [study[0]["title"]]


def test_a_parts_related_material_never_comes_from_its_own_topic(uenv):
    course_id = _big(uenv)
    first, second = _prepare(uenv, course_id, "t-0", size="compact").json()["children"]
    part = _prepare(uenv, course_id, first["id"], size="compact").json()
    assert part["related_passages"] >= 1  # the search ran and found the other topic's passage
    assert {s["section"] for s in part["sources"] if s["is_related"]} == {"Regularization"}


def test_a_plain_split_makes_parts_of_about_equal_size(uenv):
    course_id = _big(uenv)
    uenv.state.ai = UnitAI(split="gap")
    sizes = [c["estimated_tokens"] for c in _prepare(uenv, course_id, "t-0", size="compact").json()["children"]]
    assert len(sizes) == 2 and abs(sizes[0] - sizes[1]) <= 530


# ---------------------------------------------------------------- review fixes

def _units_under(uenv, course_id, node_id, size):
    """Open a topic and every part below it; return the units reached."""
    result = _prepare(uenv, course_id, node_id, size=size).json()
    if result["status"] != "split":
        return [result]
    return [unit for child in result["children"] for unit in _units_under(uenv, course_id, child["id"], size)]


def test_long_source_and_section_names_never_push_passages_out(uenv):
    long_source = "Deep Learning (Goodfellow, Bengio, Courville) - Chapter 8 Optimization for Training Deep Models.pdf"
    long_section = "8.3.1 Stochastic Gradient Descent and Its Practical Variants"
    course_id = _setup(uenv, [(long_section, [f"{n} " + "z" * 1705 for n in range(5)]), ("Other", ["Other text. " * 5])],
                       [("SGD", [long_section]), ("Other", ["Other"])], source_name=long_source)
    units_reached = _units_under(uenv, course_id, "t-0", "compact")
    assert all(unit["omitted_passages"] == 0 and unit["estimated_tokens"] <= 4000 for unit in units_reached)
    shown = [s["chunk_id"] for unit in units_reached for s in unit["sources"] if not s["is_related"]]
    with uenv.sessions() as db:
        section = db.execute(select(Section.id).where(Section.title == long_section)).scalar_one()
        expected = db.execute(select(Chunk.id).where(Chunk.section_id == section).order_by(Chunk.chunk_index)).scalars().all()
    assert shown == expected  # every passage of the topic, once, in order
    assert all(len(line) < 140 for unit in units_reached for line in unit["study_text"].splitlines() if line.startswith("### "))


def test_a_rebuilt_topic_with_the_same_revision_is_prepared_again(uenv):
    course_id = _setup(uenv, [("Loss", ["Loss measures error. " * 20]), ("Momentum", ["Momentum adds velocity. " * 20])],
                       [("Loss Functions", ["Loss"]), ("Momentum", ["Momentum"])])
    assert "Loss measures error." in _prepare(uenv, course_id, "t-0").json()["study_text"]
    with uenv.sessions() as db:  # a forced rebuild: same sources (same revision), the topic now covers other sections
        node = db.execute(select(StudyPlanNode).where(StudyPlanNode.node_id == "t-0")).scalar_one()
        momentum = db.execute(select(Section.id).where(Section.title == "Momentum")).scalar_one()
        node.title, node.metadata_json = "Momentum methods", json.dumps({"purpose": "Speed up.", "section_ids": [momentum]})
        db.commit()
    again = _prepare(uenv, course_id, "t-0").json()
    assert not again["cached"] and again["study_text"].startswith("# Study unit: Momentum methods")
    assert "Momentum adds velocity." in again["study_text"].split("## Related material")[0]


def test_parts_at_different_sizes_have_their_own_ids_and_progress(uenv):
    course_id = _setup(uenv, [("Optimization", [f"{n} " + PASSAGE for n in range(30)])], [("Gradient Descent", ["Optimization"])])
    uenv.state.ai = UnitAI(split="gap")  # plain parts, titled the same way at every size
    compact = {c["id"] for c in _prepare(uenv, course_id, "t-0", size="compact").json()["children"]}
    standard = {c["id"] for c in _prepare(uenv, course_id, "t-0", size="standard").json()["children"]}
    assert compact and standard and not compact & standard


def test_reopening_while_the_ai_is_still_unavailable_keeps_prepared_parts(uenv):
    course_id = _big(uenv)
    uenv.state.ai = UnitAI(split="quota")
    first = _prepare(uenv, course_id, "t-0", size="compact").json()["children"][0]
    assert _prepare(uenv, course_id, first["id"], size="compact").json()["status"] == "unit"
    embeds = uenv.state.ai.embeds
    again = _prepare(uenv, course_id, "t-0", size="compact").json()  # opened again: the AI is tried again, and fails again
    assert uenv.state.ai.tasks == ["outline_split", "outline_split"] and again["provisional"]
    assert again["children"][0]["id"] == first["id"] and again["children"][0]["status"] == "unit"
    assert _prepare(uenv, course_id, first["id"], size="compact").json()["cached"] and uenv.state.ai.embeds == embeds
    shown = uenv.client.post("/plan/topics/prepare", json={"course_id": course_id, "node_id": "t-0", "size": "compact", "retry": False}).json()
    assert shown["cached"] and uenv.state.ai.tasks == ["outline_split", "outline_split"]  # only shown again: no AI call


def test_a_failure_that_would_repeat_is_not_tried_again(uenv):
    course_id = _big(uenv)
    uenv.state.ai = UnitAI(split="broken")
    result = _prepare(uenv, course_id, "t-0", size="compact").json()
    assert result["status"] == "split" and "provisional" not in result
    assert _prepare(uenv, course_id, "t-0", size="compact").json()["cached"] and uenv.state.ai.tasks == ["outline_split"]


def test_a_short_part_is_a_unit_not_a_thin_topic(uenv):
    course_id = _setup(uenv, [("A", [f"{n} " + PASSAGE for n in range(5)]), ("B", ["Short note."]), ("C", [f"{n} " + PASSAGE for n in range(5)])],
                       [("Everything", ["A", "B", "C"])])
    uenv.state.ai = UnitAI(split="each")
    parts = _prepare(uenv, course_id, "t-0", size="compact").json()["children"]
    assert len(parts) == 3 and _prepare(uenv, course_id, parts[1]["id"], size="compact").json()["status"] == "unit"


def test_two_requests_for_the_same_topic_make_one_ai_call(uenv):
    import threading
    import time

    class SlowAI(UnitAI):
        def generate_json(self, prompt, task="extraction", schema=None):
            time.sleep(0.4)
            return super().generate_json(prompt, task, schema)

    course_id = _big(uenv)
    uenv.state.ai = SlowAI()
    results = []
    threads = [threading.Thread(target=lambda: results.append(_prepare(uenv, course_id, "t-0", size="compact").json())) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert uenv.state.ai.tasks == ["outline_split"] and [r["status"] for r in results] == ["split", "split"]


def test_an_interactive_request_does_not_wait_on_provider_retries(uenv):
    course_id = _big(uenv)
    uenv.state.ai.retry_max = 4
    _prepare(uenv, course_id, "t-0", size="compact")
    assert uenv.state.ai.retry_max == 0


def test_ticking_a_part_never_takes_over_an_older_title_only_progress_row(uenv):
    from app.models import StudyProgress
    course_id = _setup(uenv, [("A", [f"{n} " + PASSAGE for n in range(5)]), ("B", [f"{n} " + PASSAGE for n in range(5)])],
                       [("Neural Networks", ["A", "B"]), ("Backpropagation", ["B"])])
    with uenv.sessions() as db:  # saved before topics had ids
        db.add(StudyProgress(scope_key=f"course:{course_id}", topic_key="backpropagation", topic_label="Backpropagation", completed=True))
        db.commit()

    class NamedAI(UnitAI):
        def generate_json(self, prompt, task="extraction", schema=None):
            ids = [g["id"] for g in json.loads(re.search(r"size in tokens\): (\[.*\])\n", prompt).group(1))]
            return {"parts": [{"title": "Forward pass", "purpose": "", "group_ids": ids[:1]},
                              {"title": "Backpropagation", "purpose": "", "group_ids": ids[1:]}]}

    uenv.state.ai = NamedAI()
    part = _prepare(uenv, course_id, "t-0", size="compact").json()["children"][1]
    assert part["title"] == "Backpropagation"
    uenv.client.post("/study-progress", json={"topic": part["title"], "topic_id": part["id"], "course_id": course_id})
    progress = uenv.client.get("/study-progress", params={"course_id": course_id}).json()
    assert "Backpropagation" in progress["completed_legacy_topics"] and part["id"] in progress["completed_topic_ids"]
    uenv.client.post("/study-progress", json={"topic": "Backpropagation", "topic_id": "t-1", "course_id": course_id})
    progress = uenv.client.get("/study-progress", params={"course_id": course_id}).json()
    assert {"t-1", part["id"]} <= set(progress["completed_topic_ids"])  # the plan topic claims its own old row


def test_being_offline_is_temporary_so_the_result_is_prepared_again(uenv):
    class OfflineAI(UnitAI):
        def embed(self, texts):
            self.embeds += 1
            raise RuntimeError("[Errno 11001] getaddrinfo failed")

        def generate_json(self, prompt, task="extraction", schema=None):
            self.tasks.append(task)
            raise RuntimeError("All Gemini JSON models failed: fast: [Errno 11001] getaddrinfo failed")

    course_id = _big(uenv)
    uenv.state.ai = OfflineAI()
    assert _prepare(uenv, course_id, "t-0", size="compact").json()["provisional"]  # a plain split, tried again later
    assert _prepare(uenv, course_id, "t-1", size="compact").json()["provisional"]  # no related material, tried again later


def test_showing_a_provisional_split_again_never_calls_the_ai_even_after_a_plan_change(uenv):
    course_id = _big(uenv)
    uenv.state.ai = UnitAI(split="quota")
    children = _prepare(uenv, course_id, "t-0", size="compact").json()["children"]
    with uenv.sessions() as db:
        db.execute(select(StudyPlan)).scalar_one().revision = "r2"
        db.commit()
    uenv.state.ai = UnitAI()
    shown = uenv.client.post("/plan/topics/prepare", json={"course_id": course_id, "node_id": "t-0", "size": "compact", "retry": False}).json()
    assert uenv.state.ai.tasks == [] and shown["provisional"] and [c["id"] for c in shown["children"]] == [c["id"] for c in children]
    retried = _prepare(uenv, course_id, "t-0", size="compact").json()  # an explicit open tries the AI again
    assert uenv.state.ai.tasks == ["outline_split"] and "provisional" not in retried
