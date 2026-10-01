import json
from pathlib import Path
from app.services.ingestion import clean_caption_text, parse_file


def test_srt_cleanup(tmp_path):
    p = tmp_path / "lecture.srt"
    p.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello world.\n\n2\n00:00:04,000 --> 00:00:06,000\nRetrieval matters.\n", encoding="utf-8")
    docs = parse_file(p)
    assert len(docs) == 1
    assert "00:00" not in docs[0].text
    assert "Retrieval matters." in docs[0].text


def test_structured_json_transcript(tmp_path):
    p = tmp_path / "lecture.json"
    p.write_text(json.dumps({"name": "Lecture 1", "topics": ["RAG", "retrieval"], "transcript": "Dense retrieval finds semantically similar passages."}), encoding="utf-8")
    docs = parse_file(p)
    assert docs[0].title == "Lecture 1"
    assert "Dense retrieval" in docs[0].text


def test_python_dictionary_literal_is_safe_and_supported():
    from app.services.ingestion import parse_structured_literal
    obj = parse_structured_literal("{'topic': 'RAG', 'transcript': 'Hybrid retrieval'}")
    assert obj['topic'] == 'RAG'
    assert obj['transcript'] == 'Hybrid retrieval'
