from app.services.dedupe import mark_duplicates, shingles
from app.services.ingestion import chunk_text, strip_overlap
from tests.helpers import BM25, DENSE, RRF, add_course, add_source


def test_strip_overlap_removes_the_copied_tail():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, size=600, overlap=120)
    rebuilt = chunks[0] + " " + " ".join(strip_overlap(a, b, 160) for a, b in zip(chunks, chunks[1:]))
    assert rebuilt.split() == text.split()


def test_repeated_text_in_the_same_course_is_merged_into_the_earlier_chunk(db):
    course = add_course(db)
    first, (bm25, dense) = add_source(db, "Lecture 1", [BM25, DENSE], course)
    assert mark_duplicates(db, first) == 0

    # A recap that repeats the BM25 paragraph almost verbatim, plus new material.
    second, (recap, rrf) = add_source(db, "Lecture 2", [BM25.replace("It scores", "it scores"), RRF], course)
    assert mark_duplicates(db, second) == 1
    assert recap.duplicate_of_id == bm25.id
    assert rrf.duplicate_of_id is None
    assert {c.dedupe_status for c in (recap, rrf)} == {"done"}
    # Resume-safe: running again changes nothing.
    assert mark_duplicates(db, second) == 0


def test_other_courses_and_partial_overlap_are_not_merged(db):
    a, b = add_course(db, "A"), add_course(db, "B")
    first, _ = add_source(db, "A1", [BM25], a)
    mark_duplicates(db, first)
    other_course, (copy,) = add_source(db, "B1", [BM25], b)
    assert mark_duplicates(db, other_course) == 0
    assert copy.duplicate_of_id is None

    # Half new content is not "the same content".
    mixed, (half,) = add_source(db, "A2", [BM25 + " " + DENSE], a)
    assert mark_duplicates(db, mixed) == 0
    assert half.duplicate_of_id is None


def test_shingles_ignore_case_and_punctuation():
    assert shingles("Dense retrieval, ranks!") == shingles("dense Retrieval ranks")
