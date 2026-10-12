"""How alike two sources are, from their passages' embeddings."""
from __future__ import annotations

import json

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Chunk, Document, Source

CENTROID_SAMPLE = 200


def centroid(db: Session, source: Source) -> np.ndarray | None:
    """The source's mean embedding, unit length, over an even sample of its passages (a long playlist is
    represented by all its videos, not its first few). Cached on the source until its passages change."""
    ids = db.execute(select(Chunk.id).join(Document, Chunk.document_id == Document.id)
                     .where(Document.source_id == source.id, Chunk.chunk_kind == "leaf", Chunk.embedding_json.is_not(None))
                     .order_by(Chunk.id)).scalars().all()
    if not ids:
        return None
    key = f"v2:{len(ids)}:{ids[0]}:{ids[-1]}"
    if source.centroid_key == key and source.centroid_json:
        try:
            return np.asarray(json.loads(source.centroid_json), dtype=float)
        except (TypeError, ValueError):
            pass
    sample = [ids[int(i)] for i in np.linspace(0, len(ids) - 1, min(len(ids), CENTROID_SAMPLE)).round()]
    vectors = []
    for (embedding_json,) in db.execute(select(Chunk.embedding_json).where(Chunk.id.in_(sample))).all():
        try:
            vector = json.loads(embedding_json)
        except (TypeError, ValueError):
            continue
        if isinstance(vector, list) and vector:
            vectors.append(vector)
    if not vectors:
        return None
    sizes: dict[int, int] = {}
    for vector in vectors:
        sizes[len(vector)] = sizes.get(len(vector), 0) + 1
    size = max(sizes, key=lambda length: (sizes[length], length))  # passages embedded by another model are left out
    mean = np.mean(np.asarray([vector for vector in vectors if len(vector) == size], dtype=float), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0:
        return None
    mean = mean / norm
    source.centroid_json, source.centroid_key = json.dumps([round(float(x), 6) for x in mean]), key
    return mean


def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or len(a) != len(b):
        return None
    return float(a @ b)
