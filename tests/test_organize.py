import json
from app.models import Concept, Evidence, Group
from app.services.grouping import ConceptIndex, merge_concepts, normalize_name, organize_batch
from tests.helpers import BM25, DENSE, RRF, add_source


def answer(assignments, new_groups=(), new_concepts=()):
    return lambda prompt: {"assignments": assignments, "new_groups": list(new_groups), "new_concepts": list(new_concepts)}


def test_normalize_name_ignores_case_punctuation_and_plurals():
    assert normalize_name("Dense Retrievers!") == normalize_name("dense retriever")
    assert normalize_name("Loss") == "loss"


def test_batch_decision_names_groups_from_content_and_tags_concepts(db, scripted):
    src, chunks = add_source(db, "Lecture 1", [BM25, DENSE])
    ai = scripted(organize=answer(
        [{"excerpt": 1, "group": "Lexical retrieval", "concepts": ["BM25"]},
         {"excerpt": 2, "group": "Dense retrieval", "concepts": ["Embedding vectors", "Cosine similarity"]}],
        new_groups=[{"name": "Lexical retrieval", "description": "Term-matching ranking.", "keywords": ["bm25"]},
                    {"name": "Dense retrieval", "description": "Embedding-based ranking."}],
        new_concepts=[{"name": "BM25", "description": "A ranking function.", "aliases": ["Okapi BM25"]}],
    ))
    assert organize_batch(db, ai, src, chunks, hint="Lecture 1 of the course") == 2
    db.flush()

    lexical = db.query(Group).filter_by(name="Lexical retrieval").one()
    assert lexical.description == "Term-matching ranking."
    assert json.loads(lexical.keywords_json) == ["bm25"]
    assert chunks[0].group_id == lexical.id and chunks[0].classification_status == "done"
    assert chunks[1].group_id != lexical.id
    assert lexical.centroid_json is not None
    assert not db.query(Group).filter(Group.name.in_(["General", "Lecture 1 of the course"])).count()
    bm25 = db.query(Concept).filter_by(canonical_name="BM25").one()
    assert json.loads(bm25.aliases_json) == ["Okapi BM25"]
    assert db.query(Evidence).filter_by(concept_id=bm25.id, chunk_id=chunks[0].id).count() == 1

    # The task prompt shows the source, and one call covered the whole batch.
    assert len(ai.calls) == 1 and "[2]" in ai.calls[0][1]


def test_later_batches_reuse_groups_and_concepts_by_name_or_alias(db, scripted):
    src, chunks = add_source(db, "Lecture 1", [BM25])
    organize_batch(db, scripted(organize=answer(
        [{"excerpt": 1, "group": "Lexical retrieval", "concepts": ["BM25"]}],
        new_concepts=[{"name": "BM25", "description": "A ranking function.", "aliases": ["Okapi BM25"]}],
    )), src, chunks, None)
    db.flush()

    src2, chunks2 = add_source(db, "Lecture 2", [BM25 + " Recap.", RRF])
    ai = scripted(organize=answer([
        {"excerpt": 1, "group": "lexical Retrieval", "concepts": ["Okapi BM25"]},
        # Excerpt 2 is missing from the answer: it stays with its neighbour.
    ]))
    organize_batch(db, ai, src2, chunks2, None)
    db.flush()

    prompt = ai.calls[0][1]
    assert "- Lexical retrieval" in prompt and "concepts: BM25" in prompt
    assert db.query(Group).count() == 1
    assert chunks2[0].group_id == chunks[0].group_id == chunks2[1].group_id
    assert chunks2[1].group_confidence == 0.0
    assert db.query(Concept).count() == 1
    assert db.query(Evidence).count() == 2


def test_merge_concepts_moves_evidence_and_keeps_names_as_aliases(db, scripted):
    src, chunks = add_source(db, "Lecture", [RRF, RRF + " Again."])
    organize_batch(db, scripted(organize=answer([
        {"excerpt": 1, "group": "Rank fusion", "concepts": ["Reciprocal rank fusion"]},
        {"excerpt": 2, "group": "Rank fusion", "concepts": ["RRF", "Reciprocal rank fusion"]},
    ])), src, chunks, None)
    db.flush()
    assert db.query(Concept).count() == 2

    assert merge_concepts(db, "Reciprocal rank fusion", ["RRF"]) == 1
    kept = db.query(Concept).one()
    assert json.loads(kept.aliases_json) == ["RRF"]
    assert {e.chunk_id for e in db.query(Evidence).all()} == {chunks[0].id, chunks[1].id}
    assert db.query(Evidence).count() == 2
    assert ConceptIndex(db).find("rrf").id == kept.id


def test_offline_answer_falls_back_without_a_generic_group(db, scripted):
    src, chunks = add_source(db, "Week 3 notes", [BM25, DENSE])
    organize_batch(db, scripted(organize=lambda p: {"group_name": "Uncategorized"}), src, chunks, None)
    db.flush()
    groups = db.query(Group).all()
    assert [g.name for g in groups] == ["Week 3 notes"]
    assert {c.group_id for c in chunks} == {groups[0].id}
