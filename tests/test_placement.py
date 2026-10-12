"""Stage 6: the learner's choice of space checked against what the content is, with suggestions they decide on."""
import json
import re

from sqlalchemy import select

from app.models import Chunk, Document, MaintenanceJob, Source, SourceRelationship, SourceSpace
from app.providers.base import ProviderQuotaExceededError
from app.services import worker as worker_module
from app.services.placement import PLACEMENT_TASK, request_placement_check
from app.services.plans import PLAN_TASK, plan_revision, scope_sources
from app.services.retrieval import retrieve
from app.services.similarity import centroid
from tests.test_plans import _add_source, _course, env  # noqa: F401


class PairAI:
    """Judges source pairs by name; records which pairs it was asked about."""

    def __init__(self, relations=None, default="UNRELATED", quota=False):
        self.relations, self.default, self.quota, self.calls = relations or {}, default, quota, []

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def generate_json(self, prompt, task="extraction", schema=None):
        assert prompt.startswith("Classify these source pairs"), prompt[:80]
        if self.quota:
            self.calls.append("quota")
            raise ProviderQuotaExceededError("daily quota exhausted")
        names = {s["source_id"]: s["title"] for s in json.loads(re.search(r"Sources: (\[.*\])\nRequested pairs", prompt, re.S).group(1))}
        pairs = json.loads(re.search(r"Requested pairs: (\[.*\])", prompt).group(1))
        self.calls.append(sorted(tuple(sorted((names[a], names[b]))) for a, b in pairs))
        out = []
        for a, b in pairs:
            kind, confidence = self.relations.get(frozenset((names[a], names[b])), (self.default, 0.9))
            out.append({"source_a_id": a, "source_b_id": b, "type": kind, "confidence": confidence,
                        "rationale": f"Judged {names[a]} with {names[b]}.", "evidence": []})
        return {"relationships": out}


def _run_placement(env):
    with env.sessions() as db:
        ids = [job.id for job in db.execute(select(MaintenanceJob).where(MaintenanceJob.task == PLACEMENT_TASK,
                                                                           MaintenanceJob.status == "queued")).scalars()]
        for job_id in ids:
            db.get(MaintenanceJob, job_id).status = "running"
        db.commit()
    for job_id in ids:
        worker_module.Worker()._process_maintenance_job(job_id)
    return ids


def _library(env, **params):
    return {row["name"]: row for row in env.client.get("/sources", params=params).json()}


def _spaces(env, layout, vectors=None):
    """layout: {space name or None: [source names]}; returns ({space: id}, {source: id})."""
    vectors = vectors or {}
    with env.sessions() as db:
        spaces, sources = {}, {}
        for space, names in layout.items():
            course_id = _course(db, space).id if space else None
            spaces[space] = course_id
            for name in names:
                sources[name] = _add_source(db, course_id, name, [f"{name} topic"], vectors.get(name, (1.0, 0.0))).id
        db.commit()
    return spaces, sources


def _check(env, sources, names):
    with env.sessions() as db:
        request_placement_check(db, [sources[name] for name in names])
        db.commit()
    _run_placement(env)


SAME = ("SAME_COURSE", 0.9)


def test_a_source_filed_in_the_wrong_space_is_suggested_to_move(env):
    env.state.ai = PairAI({frozenset(("Lecture 5", "MIT Deep Learning")): SAME})
    spaces, sources = _spaces(env, {"Deep Learning": ["MIT Deep Learning"], "Statistics": ["Stats 101", "Lecture 5"]})
    _check(env, sources, ["Lecture 5"])
    suggestion = _library(env)["Lecture 5"]["placement"]
    assert suggestion["kind"] == "move" and suggestion["course_id"] == spaces["Deep Learning"] and suggestion["course_name"] == "Deep Learning"
    assert suggestion["related_source"]["name"] == "MIT Deep Learning" and "the same course as “MIT Deep Learning”" in suggestion["reason"]
    assert not suggestion["leaves_space_empty"]
    assert _library(env)["Stats 101"]["placement"] is None  # unrelated to everything else: nothing to suggest


def test_a_source_that_fits_both_spaces_is_suggested_to_be_shown_in_both(env):
    env.state.ai = PairAI({frozenset(("Lecture 5", "MIT Deep Learning")): SAME,
                           frozenset(("Lecture 5", "Stats 101")): ("COMPLEMENTARY_RESOURCE", 0.9)})
    spaces, sources = _spaces(env, {"Deep Learning": ["MIT Deep Learning"], "Statistics": ["Stats 101", "Lecture 5"]})
    _check(env, sources, ["Lecture 5"])
    assert _library(env)["Lecture 5"]["placement"]["kind"] == "also"


def test_an_unfiled_source_is_suggested_a_space_and_a_lone_source_its_course(env):
    env.state.ai = PairAI({frozenset(("Loose notes", "MIT Deep Learning")): SAME, frozenset(("Alone", "MIT Deep Learning")): SAME})
    spaces, sources = _spaces(env, {"Deep Learning": ["MIT Deep Learning"], None: ["Loose notes"], "Misc": ["Alone"]})
    _check(env, sources, ["Loose notes", "Alone"])
    library = _library(env)
    assert library["Loose notes"]["placement"]["kind"] == "add"
    assert library["Alone"]["placement"]["kind"] == "move" and library["Alone"]["placement"]["leaves_space_empty"]


def test_no_suggestion_until_the_source_is_judged_against_its_own_space(env):
    env.state.ai = PairAI()
    spaces, sources = _spaces(env, {"Deep Learning": ["MIT Deep Learning"], "Statistics": ["Stats 101", "Lecture 5"]})
    with env.sessions() as db:  # judged against another space only (e.g. by an all-sources plan)
        a, b = sorted((sources["Lecture 5"], sources["MIT Deep Learning"]))
        from app.services.learning_paths import _digest
        db.add(SourceRelationship(source_a_id=a, source_b_id=b, source_a_profile_hash=_digest({}), source_b_profile_hash=_digest({}),
                                  relation_type="SAME_COURSE", confidence=0.9, rationale="Same course."))
        db.commit()
    assert _library(env)["Lecture 5"]["placement"] is None


def test_a_judgement_made_before_a_profile_changed_is_not_used(env):
    env.state.ai = PairAI({frozenset(("Lecture 5", "MIT Deep Learning")): SAME})
    spaces, sources = _spaces(env, {"Deep Learning": ["MIT Deep Learning"], "Statistics": ["Stats 101", "Lecture 5"]})
    _check(env, sources, ["Lecture 5"])
    with env.sessions() as db:
        db.get(Source, sources["Lecture 5"]).profile_json = json.dumps({"course_title": "Something else"})
        db.commit()
    assert _library(env)["Lecture 5"]["placement"] is None


def test_a_check_asks_once_about_the_closest_sources_and_never_twice(env):
    env.state.ai = PairAI()
    layout = {"Mine": ["New", "Neighbour"], "Elsewhere": [f"Far {n}" for n in range(6)]}
    vectors = {"New": (1.0, 0.0), "Neighbour": (1.0, 0.0), **{f"Far {n}": (1.0, 0.2 * n) for n in range(6)}}
    spaces, sources = _spaces(env, layout, vectors)
    _check(env, sources, ["New"])
    assert len(env.state.ai.calls) == 1
    asked = {name for pair in env.state.ai.calls[0] for name in pair} - {"New"}
    assert asked == {"Neighbour", "Far 0", "Far 1", "Far 2", "Far 3"}  # its space, and the four most similar elsewhere
    _check(env, sources, ["New"])
    assert len(env.state.ai.calls) == 1  # every pair is judged already: no call


def test_a_quota_error_skips_the_rest_of_the_run_without_more_calls(env):
    env.state.ai = PairAI(quota=True)
    spaces, sources = _spaces(env, {"A": ["One", "Two"], "B": ["Three"]})
    _check(env, sources, ["One", "Two", "Three"])
    assert env.state.ai.calls == ["quota"]
    with env.sessions() as db:
        job = db.execute(select(MaintenanceJob).where(MaintenanceJob.task == PLACEMENT_TASK)).scalar_one()
        errors = json.loads(job.errors_json)
        assert len(errors) == 3 and sum("Skipped" in message for message in errors.values()) == 2
        assert db.query(SourceRelationship).count() == 0


def test_requests_are_collected_in_one_waiting_job(env):
    spaces, sources = _spaces(env, {"A": ["One", "Two"]})
    with env.sessions() as db:
        request_placement_check(db, [sources["One"]])
        request_placement_check(db, [sources["One"], sources["Two"]])
        db.commit()
        jobs = db.execute(select(MaintenanceJob).where(MaintenanceJob.task == PLACEMENT_TASK)).scalars().all()
        assert len(jobs) == 1 and [item["id"] for item in json.loads(jobs[0].items_json)] == [sources["One"], sources["Two"]]


def test_moving_on_a_suggestion_refiles_the_source_and_updates_both_plans(env):
    env.state.ai = PairAI({frozenset(("Lecture 5", "MIT Deep Learning")): SAME})
    spaces, sources = _spaces(env, {"Deep Learning": ["MIT Deep Learning"], "Statistics": ["Stats 101", "Lecture 5"]})
    _check(env, sources, ["Lecture 5"])
    answer = env.client.post(f"/sources/{sources['Lecture 5']}/placement", json={"action": "move", "course_id": spaces["Deep Learning"]})
    assert answer.status_code == 200 and answer.json()["course_id"] == spaces["Deep Learning"]
    row = _library(env)["Lecture 5"]
    assert row["course_id"] == spaces["Deep Learning"] and row["placement"] is None and "matches sources there" in row["assignment_reason"]
    with env.sessions() as db:
        scopes = {json.loads(job.items_json)[0]["scope"] for job in db.execute(select(MaintenanceJob).where(MaintenanceJob.task == PLAN_TASK)).scalars()}
        assert {f"course:{spaces['Deep Learning']}", f"course:{spaces['Statistics']}"} <= scopes


def test_showing_in_both_puts_the_source_in_the_other_space_everywhere(env):
    env.state.ai = PairAI({frozenset(("Lecture 5", "MIT Deep Learning")): SAME,
                           frozenset(("Lecture 5", "Stats 101")): ("COMPLEMENTARY_RESOURCE", 0.9)})
    spaces, sources = _spaces(env, {"Deep Learning": ["MIT Deep Learning"], "Statistics": ["Stats 101", "Lecture 5"]})
    _check(env, sources, ["Lecture 5"])
    deep = spaces["Deep Learning"]
    with env.sessions() as db:
        before = plan_revision(db, deep)
    answer = env.client.post(f"/sources/{sources['Lecture 5']}/placement", json={"action": "show_in_both", "course_id": deep}).json()
    assert answer["course_id"] == spaces["Statistics"] and answer["also_in"] == [{"course_id": deep, "course_name": "Deep Learning"}]
    assert set(_library(env, course_id=deep)) == {"MIT Deep Learning", "Lecture 5"}  # the space's library shows it
    assert _library(env)["Lecture 5"]["placement"] is None
    with env.sessions() as db:
        assert {s.name for s in scope_sources(db, deep)} == {"MIT Deep Learning", "Lecture 5"}  # its study plan includes it
        assert plan_revision(db, deep) != before
        found = {db.get(Document, chunk.document_id).source_id for chunk, _ in retrieve(db, env.state.ai, "topic", course_id=deep, use_reranker=False)}
        assert sources["Lecture 5"] in found  # and search in that space reaches it
    removed = env.client.delete(f"/sources/{sources['Lecture 5']}/spaces/{deep}")
    assert removed.status_code == 200 and set(_library(env, course_id=deep)) == {"MIT Deep Learning"}
    assert _library(env)["Lecture 5"]["placement"] is None  # and it is not suggested there again
    assert env.client.delete(f"/sources/{sources['Lecture 5']}/spaces/{deep}").status_code == 404


def test_keeping_a_source_where_it_is_stops_that_suggestion(env):
    env.state.ai = PairAI({frozenset(("Lecture 5", "MIT Deep Learning")): SAME})
    spaces, sources = _spaces(env, {"Deep Learning": ["MIT Deep Learning"], "Statistics": ["Stats 101", "Lecture 5"]})
    _check(env, sources, ["Lecture 5"])
    env.client.post(f"/sources/{sources['Lecture 5']}/placement", json={"action": "keep", "course_id": spaces["Deep Learning"]})
    row = _library(env)["Lecture 5"]
    assert row["placement"] is None and row["course_id"] == spaces["Statistics"]


def test_an_unfiled_source_cannot_be_shown_in_a_second_space(env):
    spaces, sources = _spaces(env, {"Deep Learning": ["MIT Deep Learning"], None: ["Loose notes"]})
    answer = env.client.post(f"/sources/{sources['Loose notes']}/placement", json={"action": "show_in_both", "course_id": spaces["Deep Learning"]})
    assert answer.status_code == 422
    assert env.client.post("/sources/999/placement", json={"action": "keep", "course_id": spaces["Deep Learning"]}).status_code == 404


def test_removing_a_source_removes_where_it_is_shown_and_updates_those_plans(env):
    spaces, sources = _spaces(env, {"A": ["Keep me", "Shared"], "B": ["Other"]})
    with env.sessions() as db:
        db.add(SourceSpace(source_id=sources["Shared"], course_id=spaces["B"]))
        db.commit()
    assert env.client.delete(f"/sources/{sources['Shared']}").status_code == 200
    with env.sessions() as db:
        assert db.query(SourceSpace).count() == 0
        scopes = {json.loads(job.items_json)[0]["scope"] for job in db.execute(select(MaintenanceJob).where(MaintenanceJob.task == PLAN_TASK)).scalars()}
        assert {f"course:{spaces['A']}", f"course:{spaces['B']}"} <= scopes


def test_a_long_source_is_represented_by_all_its_parts_and_the_centroid_is_cached(env):
    with env.sessions() as db:
        course = _course(db, "Long")
        source = _add_source(db, course.id, "Playlist", [f"Video {n}" for n in range(400)], (1.0, 0.0))
        for chunk in db.execute(select(Chunk).join(Document).where(Document.source_id == source.id).order_by(Chunk.id)).scalars().all()[200:]:
            chunk.embedding_json = json.dumps([0.0, 1.0])  # the second half of the playlist is about something else
        db.flush()
        vector = centroid(db, source)
        assert abs(vector[0] - vector[1]) < 0.1  # both halves count, not just the first 200 passages
        db.commit()
        first = db.execute(select(Chunk).join(Document).where(Document.source_id == source.id).order_by(Chunk.id)).scalars().first()
        first.embedding_json = json.dumps([0.0, 1.0])  # same passages: the cached value is used
        assert max(abs(a - b) for a, b in zip(centroid(db, source), vector)) < 1e-5  # recomputing would give about 0.700, not 0.707


def test_a_misfiled_source_never_pulls_its_counterpart_into_the_wrong_space(env):
    env.state.ai = PairAI({frozenset(("Lecture 5", "MIT Deep Learning")): SAME})
    spaces, sources = _spaces(env, {"Deep Learning": ["MIT Deep Learning"], "Statistics": ["Stats 101", "Lecture 5"]})
    _check(env, sources, ["Lecture 5", "MIT Deep Learning", "Stats 101"])
    library = _library(env)
    assert library["Lecture 5"]["placement"]["kind"] == "move"
    assert library["MIT Deep Learning"]["placement"] is None  # Lecture 5 does not belong in Statistics, so it is no evidence for it


def test_two_lone_sources_of_one_course_get_one_suggestion_not_two(env):
    env.state.ai = PairAI({frozenset(("Part A", "Part B")): SAME})
    spaces, sources = _spaces(env, {"First": ["Part A"], "Second": ["Part B"]})
    _check(env, sources, ["Part A", "Part B"])
    library = _library(env)
    assert library["Part A"]["placement"] is None
    assert library["Part B"]["placement"]["kind"] == "move" and library["Part B"]["placement"]["course_name"] == "First"


# ---------------------------------------------------------------- review fixes

def test_a_lone_source_is_not_pulled_toward_a_space_where_its_partner_is_only_a_companion(env):
    env.state.ai = PairAI({frozenset(("Lecture 5", "MIT Deep Learning")): SAME,
                           frozenset(("Lecture 5", "Stats 101")): ("COMPLEMENTARY_RESOURCE", 0.9)})
    spaces, sources = _spaces(env, {"Deep Learning": ["MIT Deep Learning"], "Statistics": ["Stats 101", "Lecture 5"]})
    _check(env, sources, ["Lecture 5", "MIT Deep Learning", "Stats 101"])
    library = _library(env)
    assert library["Lecture 5"]["placement"]["kind"] == "also"
    assert library["MIT Deep Learning"]["placement"] is None


def test_a_newcomer_alone_in_a_fresh_space_is_asked_to_join_not_the_established_course(env):
    env.state.ai = PairAI({frozenset(("Newcomer", "Lecture B")): SAME, frozenset(("Lecture B", "Lecture M")): SAME})
    spaces, sources = _spaces(env, {"Course": ["Lecture B", "Lecture M"], "Fresh": ["Newcomer"]})
    _check(env, sources, ["Newcomer", "Lecture B", "Lecture M"])
    library = _library(env)
    assert library["Newcomer"]["placement"]["kind"] == "move" and library["Newcomer"]["placement"]["course_name"] == "Course"
    assert library["Lecture B"]["placement"] is None and library["Lecture M"]["placement"] is None


def test_a_source_only_shown_in_a_space_is_never_evidence_for_others_to_join_it(env):
    env.state.ai = PairAI({frozenset(("X", "N")): SAME, frozenset(("X", "Y")): ("COMPLEMENTARY_RESOURCE", 0.9),
                           frozenset(("Y", "M")): SAME})
    spaces, sources = _spaces(env, {"DL": ["X", "N"], "Stats": ["Y", "M"]})
    _check(env, sources, ["X", "N", "Y", "M"])
    env.client.post(f"/sources/{sources['X']}/placement", json={"action": "show_in_both", "course_id": spaces["Stats"]})
    assert _library(env)["N"]["placement"] is None


def test_a_check_compares_with_filed_sources_not_with_unfiled_ones(env):
    env.state.ai = PairAI({frozenset(("U1", "Y")): SAME, frozenset(("Y", "Z")): SAME})
    vectors = {**{f"U{n}": (1.0, 0.0) for n in range(1, 6)}, "Y": (1.0, 0.5), "Z": (1.0, 0.6)}
    spaces, sources = _spaces(env, {None: [f"U{n}" for n in range(1, 6)], "Course": ["Y", "Z"]}, vectors)
    _check(env, sources, ["U1", "Y"])
    assert {name for pair in env.state.ai.calls[0] for name in pair} == {"U1", "Y", "Z"}  # not the similar unfiled U2..U5
    assert env.state.ai.calls[1] == [("Y", "Z")]  # a filed source is compared with filed sources only
    assert _library(env)["U1"]["placement"]["kind"] == "add"


def test_an_even_sample_covers_a_source_of_any_length(env):
    with env.sessions() as db:
        source = _add_source(db, _course(db, "Long").id, "Playlist", [f"Video {n}" for n in range(399)], (1.0, 0.0))
        for chunk in db.execute(select(Chunk).join(Document).where(Document.source_id == source.id).order_by(Chunk.id)).scalars().all()[200:]:
            chunk.embedding_json = json.dumps([0.0, 1.0])
        db.flush()
        vector = centroid(db, source)
        assert abs(vector[0] - vector[1]) < 0.1


def test_after_the_newer_lone_source_is_kept_the_older_one_is_offered_to_join_it(env):
    env.state.ai = PairAI({frozenset(("Part A", "Part B")): SAME})
    spaces, sources = _spaces(env, {"First": ["Part A"], "Second": ["Part B"]})
    _check(env, sources, ["Part A", "Part B"])
    env.client.post(f"/sources/{sources['Part B']}/placement", json={"action": "keep", "course_id": spaces["First"]})
    library = _library(env)
    assert library["Part B"]["placement"] is None
    assert library["Part A"]["placement"]["kind"] == "move" and library["Part A"]["placement"]["course_name"] == "Second"


def test_unfiling_a_source_stops_showing_it_in_other_spaces(env):
    spaces, sources = _spaces(env, {"S1": ["X", "Other"], "S2": ["Y"]})
    with env.sessions() as db:
        db.add(SourceSpace(source_id=sources["X"], course_id=spaces["S2"]))
        db.commit()
    env.client.patch(f"/sources/{sources['X']}/course", json={"course_id": None})
    row = _library(env)["X"]
    assert row["course_id"] is None and row["also_in"] == []
    assert set(_library(env, course_id=spaces["S2"])) == {"Y"}
    with env.sessions() as db:
        scopes = {json.loads(job.items_json)[0]["scope"] for job in db.execute(select(MaintenanceJob).where(MaintenanceJob.task == PLAN_TASK)).scalars()}
        assert {f"course:{spaces['S1']}", f"course:{spaces['S2']}"} <= scopes


def test_a_source_filed_automatically_queues_its_new_spaces_plan_and_a_placement_check(env, tmp_path):
    from app.services.queue import enqueue
    from tests.test_reliability import GOOD_PROFILE, Scripted
    with env.sessions() as db:
        course = _course(db, "Machine Learning")
        db.commit()
        course_id = course.id
    path = tmp_path / "notes.txt"
    path.write_text("Gradient descent lowers the loss.\n\nThe learning rate sets the step size.", encoding="utf-8")
    with env.sessions() as db:
        source = Source(name="notes.txt", source_type="file", local_path=str(path), status="queued", assignment_mode="auto")
        db.add(source)
        db.flush()
        job_id = enqueue(db, source.id).id
        db.commit()
        source_id = source.id
    answer = {**GOOD_PROFILE, "course_id": course_id, "new_course_name": "", "confidence": 0.9, "reason": "It is about ML."}
    env.state.ai = Scripted(answers={"classification": answer})
    worker_module.Worker().process_job(job_id)
    with env.sessions() as db:
        assert db.get(Source, source_id).course_id == course_id
        scopes = {json.loads(job.items_json)[0]["scope"] for job in db.execute(select(MaintenanceJob).where(MaintenanceJob.task == PLAN_TASK)).scalars()}
        assert f"course:{course_id}" in scopes
        assert db.execute(select(MaintenanceJob).where(MaintenanceJob.task == PLACEMENT_TASK)).scalar_one_or_none() is not None


def test_a_rate_limit_or_a_bad_key_stops_the_placement_run(env):
    from app.providers.base import ProviderAuthenticationError, ProviderRateLimitError
    for error in (ProviderRateLimitError("429"), ProviderAuthenticationError("401")):
        class Refusing(PairAI):
            def generate_json(self, prompt, task="extraction", schema=None, error=error):
                self.calls.append(type(error).__name__)
                raise error
        env.state.ai = Refusing()
        spaces, sources = _spaces(env, {f"A {type(error).__name__}": [f"One {type(error).__name__}", f"Two {type(error).__name__}"]})
        names = [f"One {type(error).__name__}", f"Two {type(error).__name__}"]
        with env.sessions() as db:
            for job in db.query(MaintenanceJob).filter_by(task=PLACEMENT_TASK):
                job.status = "done"  # the previous round's job
            db.commit()
        _check(env, sources, names)
        assert env.state.ai.calls == [type(error).__name__]
        with env.sessions() as db:
            job = db.execute(select(MaintenanceJob).where(MaintenanceJob.task == PLACEMENT_TASK).order_by(MaintenanceJob.id.desc())).scalars().first()
            assert len(json.loads(job.errors_json)) == 2


def test_a_space_search_uses_prerequisites_from_sources_only_shown_there(env):
    from app.models import Concept, ConceptRelation, Evidence
    from app.services.knowledge_graph import prerequisite_evidence_chunks
    spaces, sources = _spaces(env, {"A": ["Shared"], "B": ["Local"]})
    with env.sessions() as db:
        db.add(SourceSpace(source_id=sources["Shared"], course_id=spaces["B"]))
        chunk = db.execute(select(Chunk).join(Document).where(Document.source_id == sources["Shared"])).scalars().first()
        target, prerequisite = Concept(canonical_name="Backprop"), Concept(canonical_name="Chain rule")
        db.add_all([target, prerequisite])
        db.flush()
        db.add(Evidence(concept_id=prerequisite.id, chunk_id=chunk.id))
        db.add(ConceptRelation(from_concept_id=prerequisite.id, to_concept_id=target.id, relation_type="prerequisite",
                               evidence_json=json.dumps({"chunk_ids": [chunk.id]})))
        db.commit()
        found_a = prerequisite_evidence_chunks(db, ["Backprop"], 2, course_id=spaces["A"])
        found_b = prerequisite_evidence_chunks(db, ["Backprop"], 2, course_id=spaces["B"])
        assert [c.id for c, _, _ in found_a] == [c.id for c, _, _ in found_b] == [chunk.id]


def test_moving_instead_of_showing_in_both_does_not_offer_the_old_space_back(env):
    env.state.ai = PairAI({frozenset(("Lecture 5", "MIT Deep Learning")): SAME,
                           frozenset(("Lecture 5", "Stats 101")): ("COMPLEMENTARY_RESOURCE", 0.9)})
    spaces, sources = _spaces(env, {"Deep Learning": ["MIT Deep Learning"], "Statistics": ["Stats 101", "Lecture 5"]})
    _check(env, sources, ["Lecture 5"])
    env.client.post(f"/sources/{sources['Lecture 5']}/placement", json={"action": "move", "course_id": spaces["Deep Learning"]})
    assert _library(env)["Lecture 5"]["placement"] is None


def test_keep_here_stops_suggestions_to_move_anywhere_while_it_stays(env):
    env.state.ai = PairAI({frozenset(("Lecture 5", "MIT Deep Learning")): SAME, frozenset(("Lecture 5", "ML Course")): SAME})
    spaces, sources = _spaces(env, {"Deep Learning": ["MIT Deep Learning"], "ML": ["ML Course"], "Statistics": ["Stats 101", "Lecture 5"]})
    _check(env, sources, ["Lecture 5"])
    first = _library(env)["Lecture 5"]["placement"]
    env.client.post(f"/sources/{sources['Lecture 5']}/placement", json={"action": "keep", "course_id": first["course_id"]})
    assert _library(env)["Lecture 5"]["placement"] is None


def test_a_space_with_a_source_still_processing_is_not_called_empty(env):
    env.state.ai = PairAI({frozenset(("Alone", "MIT Deep Learning")): SAME})
    spaces, sources = _spaces(env, {"Deep Learning": ["MIT Deep Learning"], "Misc": ["Alone"]})
    with env.sessions() as db:
        db.add(Source(name="Pending", source_type="pdf", status="processing", course_id=spaces["Misc"]))
        db.commit()
    _check(env, sources, ["Alone"])
    suggestion = _library(env)["Alone"]["placement"]
    assert suggestion["kind"] == "move" and not suggestion["leaves_space_empty"]


# ---------------------------------------------------------------- verification fixes

def _judged(env, sources, pairs):
    """Save judgements directly: pairs {(name, name): (type, confidence)}."""
    from app.services.learning_paths import _digest
    with env.sessions() as db:
        for (a_name, b_name), (kind, confidence) in pairs.items():
            a, b = sorted((sources[a_name], sources[b_name]))
            db.add(SourceRelationship(source_a_id=a, source_b_id=b, source_a_profile_hash=_digest({}), source_b_profile_hash=_digest({}),
                                      relation_type=kind, confidence=confidence, rationale="Judged."))
        db.commit()


def test_a_lone_source_and_an_established_one_never_point_at_each_others_space_through_a_third(env):
    spaces, sources = _spaces(env, {"C1": ["s5"], "C2": ["s1", "s3"]})
    _judged(env, sources, {("s1", "s3"): ("SAME_COURSE", 0.7), ("s1", "s5"): ("SAME_LEARNING_PATH", 1.0),
                           ("s3", "s5"): ("COMPLEMENTARY_RESOURCE", 0.9)})
    library = _library(env)
    assert library["s5"]["placement"]["kind"] == "move" and library["s5"]["placement"]["course_name"] == "C2"
    assert library["s1"]["placement"] is None and library["s3"]["placement"] is None


def test_a_lone_source_and_an_established_one_are_never_swapped(env):
    spaces, sources = _spaces(env, {"C1": ["s2"], "C2": ["s1", "s3", "s4"]})
    _judged(env, sources, {("s1", "s3"): ("SAME_COURSE", 0.7), ("s1", "s4"): ("COMPLEMENTARY_RESOURCE", 0.75),
                           ("s2", "s3"): ("SAME_COURSE", 0.6), ("s2", "s4"): ("SAME_COURSE", 0.75)})
    library = _library(env)
    to_c1 = [name for name in ("s1", "s3", "s4") if (library[name]["placement"] or {}).get("course_name") == "C1"]
    assert not (library["s2"]["placement"] and to_c1)


def test_when_the_established_partner_refuses_the_lone_space_the_lone_source_is_offered_to_join(env):
    spaces, sources = _spaces(env, {"Course": ["O", "W"], "Fresh": ["S"]})
    _judged(env, sources, {("O", "W"): ("SAME_COURSE", 0.6), ("S", "O"): ("SAME_COURSE", 0.9)})
    library = _library(env)
    assert library["S"]["placement"] is None and library["O"]["placement"]["kind"] == "also"
    env.client.post(f"/sources/{sources['O']}/placement", json={"action": "keep", "course_id": spaces["Fresh"]})
    library = _library(env)
    assert library["O"]["placement"] is None
    assert library["S"]["placement"]["kind"] == "move" and library["S"]["placement"]["course_name"] == "Course"


def test_a_check_never_compares_with_spaces_the_source_is_already_shown_in(env):
    env.state.ai = PairAI({frozenset(("X", "Target")): SAME})
    vectors = {"X": (1.0, 0.0), "N": (1.0, 0.0), **{f"G{n}": (1.0, 0.0) for n in range(4)}, "Target": (1.0, 0.3)}
    spaces, sources = _spaces(env, {"Mine": ["X", "N"], "Shown": [f"G{n}" for n in range(4)], "Other": ["Target"]}, vectors)
    with env.sessions() as db:
        db.add(SourceSpace(source_id=sources["X"], course_id=spaces["Shown"]))
        db.commit()
    _check(env, sources, ["X"])
    assert "Target" in {name for pair in env.state.ai.calls[0] for name in pair}


def test_adding_a_source_again_to_a_space_it_is_shown_in_is_not_a_conflict(env):
    from app.main import _assignment_conflict
    spaces, sources = _spaces(env, {"A": ["X"], "B": ["Y"], "C": ["Z"]})
    with env.sessions() as db:
        db.add(SourceSpace(source_id=sources["X"], course_id=spaces["B"]))
        db.flush()
        existing = db.get(Source, sources["X"])
        assert not _assignment_conflict(db, existing, "explicit_course", spaces["A"])
        assert not _assignment_conflict(db, existing, "explicit_course", spaces["B"])
        assert _assignment_conflict(db, existing, "explicit_course", spaces["C"])
        assert _assignment_conflict(db, existing, "explicit_unassigned", None)


def test_suggestions_never_contradict_each_other_across_random_libraries(env):
    """Seeded random libraries: suggestions are checked against rules that must always hold."""
    import random
    from app.models import Course
    from app.services.learning_paths import _digest
    from app.services.placement import FIT, BELONGS, suggestions
    kinds = ["SAME_COURSE", "SAME_LEARNING_PATH", "COMPLEMENTARY_RESOURCE", "SAME_DOMAIN_DIFFERENT_COURSE", "UNRELATED"]
    rng = random.Random(20261012)
    for _ in range(150):
        with env.sessions() as db:
            courses = [Course(name=f"Space {rng.random()}") for _ in range(rng.randint(2, 4))]
            db.add_all(courses)
            db.flush()
            made = []
            for course in courses:
                for _ in range(rng.randint(1, 3)):
                    made.append(Source(name=f"s{rng.random()}", source_type="pdf", status="done", course_id=course.id))
            if rng.random() < 0.3:
                made.append(Source(name="unfiled", source_type="pdf", status="done"))
            db.add_all(made)
            db.flush()
            for i, a in enumerate(made):
                for b in made[i + 1:]:
                    if rng.random() < 0.6:
                        low, high = sorted((a.id, b.id))
                        db.add(SourceRelationship(source_a_id=low, source_b_id=high, source_a_profile_hash=_digest({}),
                                                  source_b_profile_hash=_digest({}), relation_type=rng.choice(kinds),
                                                  confidence=round(rng.uniform(0.5, 1.0), 2), rationale="Judged."))
            db.flush()
            out = suggestions(db, made)
            by_id = {s.id: s for s in made}
            filed = {}
            for s in made:
                if s.course_id is not None:
                    filed.setdefault(s.course_id, set()).add(s.id)
            for source_id, suggestion in out.items():
                if not suggestion:
                    continue
                source = by_id[source_id]
                assert suggestion["course_id"] != source.course_id  # never its own space
                assert suggestion["related_source"]["id"] in filed[suggestion["course_id"]]  # evidence filed there
                rel = suggestion["relation"]
                assert BELONGS.get(rel, 0) * suggestion["confidence"] >= FIT - 1e-9
                lone = source.course_id is not None and filed[source.course_id] == {source_id}
                if lone:  # a lone source asked to join a space: nobody there is asked to come to the lone one's space
                    for other_id in filed[suggestion["course_id"]]:
                        other = out.get(other_id)
                        assert not (other and other["course_id"] == source.course_id), (suggestion, other)
            db.rollback()


# ---------------------------------------------------------------- background organization

def test_the_library_is_organized_in_the_background_without_being_asked(env, monkeypatch):
    from app.config import settings
    from app.services.organize import BACKGROUND_TASK
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    spaces, sources = _spaces(env, {"A": ["One", "Two"]})
    with env.sessions() as db:  # profiles made, passages grouped and read: only the placement check is pending
        from app.models import Group
        from app.services.concepts import CONCEPT_PASS
        for source in db.query(Source).all():
            source.profile_json = json.dumps({"schema_version": 2, "course_title": source.name})
        group = Group(name="General")
        db.add(group)
        db.flush()
        for chunk in db.query(Chunk).all():
            chunk.group_id, chunk.classification_status, chunk.concept_pass = group.id, "done", CONCEPT_PASS
        db.commit()
    worker = worker_module.Worker()
    worker._maybe_organize()  # just started: it waits a minute first
    with env.sessions() as db:
        assert db.query(MaintenanceJob).filter_by(task=BACKGROUND_TASK).count() == 0
    worker._next_organize = 0.0
    worker._maybe_organize()
    with env.sessions() as db:
        job = db.query(MaintenanceJob).filter_by(task=BACKGROUND_TASK).one()
        items = json.loads(job.items_json)
        assert {item["kind"] for item in items} == {"placement"}  # never compared with each other yet
        job.status, job.progress_current = "done", job.progress_total
        a, b = sorted(sources.values())  # now they have been compared
        from app.services.learning_paths import _digest
        db.add(SourceRelationship(source_a_id=a, source_b_id=b, source_a_profile_hash=_digest({}), source_b_profile_hash=_digest({}),
                                  relation_type="SAME_COURSE", confidence=0.9, rationale="Judged."))
        db.commit()
    worker._next_organize = 0.0
    worker._maybe_organize()
    with env.sessions() as db:  # nothing left to do: no new run
        assert db.query(MaintenanceJob).filter_by(task=BACKGROUND_TASK).count() == 1


def test_no_background_organization_without_an_ai_key(env, monkeypatch):
    from app.config import settings
    from app.services.organize import ORGANIZE_TASKS
    monkeypatch.setattr(settings, "gemini_api_key", "")
    spaces, sources = _spaces(env, {"A": ["One", "Two"]})
    worker = worker_module.Worker()
    worker._next_organize = 0.0
    worker._maybe_organize()
    with env.sessions() as db:
        assert db.query(MaintenanceJob).filter(MaintenanceJob.task.in_(ORGANIZE_TASKS)).count() == 0


def _background_jobs(env):
    from app.services.organize import BACKGROUND_TASK
    with env.sessions() as db:
        return db.query(MaintenanceJob).filter_by(task=BACKGROUND_TASK).order_by(MaintenanceJob.id).all()


def _run_job(env, job_id):
    with env.sessions() as db:
        db.get(MaintenanceJob, job_id).status = "running"
        db.commit()
    worker_module.Worker()._process_maintenance_job(job_id)


def test_a_background_run_never_repeats_work_that_failed_while_nothing_changed(env, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(worker_module, "extract_batch_concepts", lambda db, ai, source, chunk_ids: None)  # a batch that keeps failing
    relations = []
    monkeypatch.setattr(worker_module, "extract_source_relations", lambda db, ai, source_id, **kwargs: relations.append(source_id))
    spaces, sources = _spaces(env, {"A": ["Only"]})
    with env.sessions() as db:
        for source in db.query(Source).all():
            source.profile_json = json.dumps({"schema_version": 2, "course_title": source.name})
        db.commit()
    worker = worker_module.Worker()
    worker._next_organize = 0.0
    worker._maybe_organize()
    jobs = _background_jobs(env)
    assert len(jobs) == 1 and {item["kind"] for item in json.loads(jobs[0].items_json)} >= {"concepts"}
    _run_job(env, jobs[0].id)
    assert relations == []  # nothing new was read: no relation extraction, no AI calls
    worker._next_organize = 0.0
    worker._maybe_organize()
    assert len(_background_jobs(env)) == 1  # the same work on the same state is not tried again


def test_work_the_learner_waits_for_goes_before_organizing(env, monkeypatch):
    from app.services.organize import queue_organization
    from app.services.plans import request_plan_update
    spaces, sources = _spaces(env, {"A": ["One", "Two"]})
    with env.sessions() as db:
        organize, _ = queue_organization(db, include_unassigned=False, automatic=True)
        plan = request_plan_update(db, spaces["A"])
        db.commit()
        organize_id, plan_id = organize.id, plan.id
    assert organize_id < plan_id
    worker = worker_module.Worker()
    assert worker._claim_maintenance_job() == plan_id  # queued later, claimed first
    with env.sessions() as db:
        db.get(MaintenanceJob, plan_id).status = "queued"  # waiting again
        db.get(MaintenanceJob, organize_id).status = "running"
        db.commit()
    worker._process_maintenance_job(organize_id)
    with env.sessions() as db:
        job = db.get(MaintenanceJob, organize_id)
        assert job.status == "queued" and job.progress_current == 0  # it made way, and resumes from where it was


def test_a_source_waiting_in_a_background_run_can_still_be_removed(env):
    from app.services.organize import queue_organization
    spaces, sources = _spaces(env, {"A": ["One", "Two"]})
    with env.sessions() as db:
        job, _ = queue_organization(db, include_unassigned=False, automatic=True)
        assert any(item["id"] == sources["One"] for item in json.loads(job.items_json))
        db.commit()
    assert env.client.delete(f"/sources/{sources['One']}").status_code == 200


def test_a_plan_built_without_an_ai_model_is_rebuilt_once_a_key_is_added(env, monkeypatch):
    from app.config import settings
    spaces, sources = _spaces(env, {"A": ["One"]})
    monkeypatch.setattr(settings, "gemini_api_key", "")
    with env.sessions() as db:
        offline = plan_revision(db, spaces["A"])
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    with env.sessions() as db:
        assert plan_revision(db, spaces["A"]) != offline
