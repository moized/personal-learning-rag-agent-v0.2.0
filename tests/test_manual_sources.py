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


def test_youtube_url_parser_rejects_unrelated_hosts_and_bad_video_ids():
    from app.services.ingestion import list_youtube_videos, youtube_video_id

    assert youtube_video_id("https://www.youtube.com/watch?v=BaW_jenozKc") == "BaW_jenozKc"
    assert youtube_video_id("https://youtube.com.evil.example/watch?v=BaW_jenozKc") is None
    assert youtube_video_id("https://www.youtube.com/watch?v=bad") is None
    import pytest
    with pytest.raises(ValueError, match="YouTube video or playlist"):
        list_youtube_videos("https://example.com/watch?list=PL123")


def test_youtube_transcript_extraction_failure_is_raised(monkeypatch):
    from youtube_transcript_api import YouTubeTranscriptApi
    from app.services.ingestion import fetch_youtube_transcript

    def fail_fetch(self, video_id, languages):
        raise RuntimeError("captions blocked by network")

    monkeypatch.setattr(YouTubeTranscriptApi, "fetch", fail_fetch)
    import pytest
    with pytest.raises(RuntimeError, match="Transcript unavailable.*captions blocked by network"):
        fetch_youtube_transcript("BaW_jenozKc", languages=("en",))
