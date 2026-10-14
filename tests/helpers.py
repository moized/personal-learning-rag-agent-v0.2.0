import json
from app.models import Chunk, Course, Document, Source
from app.providers.local import LocalProvider

_local = LocalProvider(64)


def add_source(db, name, paragraphs, course=None, embed=True):
    """A source whose document holds one chunk per paragraph (no worker needed)."""
    src = Source(name=name, source_type="transcript", status="processing", course_id=course.id if course else None)
    db.add(src); db.flush()
    doc = Document(source_id=src.id, external_id=name, title=name, section=None, status="processing")
    db.add(doc); db.flush()
    chunks = []
    for i, text in enumerate(paragraphs):
        chunk = Chunk(document_id=doc.id, chunk_index=i, text=text, token_estimate=len(text) // 4)
        if embed:
            chunk.embedding_json = json.dumps(_local.embed([text])[0])
            chunk.embedding_status = "done"
        db.add(chunk)
        chunks.append(chunk)
    db.flush()
    return src, chunks


def add_course(db, name="Retrieval"):
    course = Course(name=name)
    db.add(course); db.flush()
    return course


BM25 = ("BM25 is a lexical ranking function. It scores documents by term frequency, inverse document "
        "frequency and document length normalization, which makes it strong for exact technical terms.")
DENSE = ("Dense retrieval encodes queries and passages as embedding vectors and ranks passages by cosine "
         "similarity, which captures meaning even when the wording differs.")
RRF = ("Reciprocal rank fusion combines several ranked lists by summing one over k plus rank for every list "
       "a passage appears in, so passages ranked well by both retrievers rise to the top.")
