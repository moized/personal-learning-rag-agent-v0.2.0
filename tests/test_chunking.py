from app.services.ingestion import chunk_text


def test_chunking_overlap_and_coverage():
    text = " ".join([f"word{i}" for i in range(250)])
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    assert chunks[0]
    assert chunks[-1].endswith("word249")
