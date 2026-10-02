from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Concept, ConceptRelation, Group
from app.services.knowledge_graph import prerequisite_chain
from app.services.retrieval import rrf_fuse_many


def test_prerequisite_chain_walks_backward():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as db:
        group = Group(name="AI")
        a = Concept(canonical_name="Linear Algebra", description="foundations", group_id=None)
        b = Concept(canonical_name="Eigenvalues", description="eigenvalues")
        c = Concept(canonical_name="PCA", description="principal component analysis")
        db.add_all([group, a, b, c])
        db.flush()
        db.add_all([
            ConceptRelation(from_concept_id=a.id, to_concept_id=b.id, relation_type="prerequisite", confidence=0.9),
            ConceptRelation(from_concept_id=b.id, to_concept_id=c.id, relation_type="prerequisite", confidence=0.9),
        ])
        db.commit()

        chain = prerequisite_chain(db, [c.id], depth=2)
        assert [concept.canonical_name for concept, _ in chain] == ["Eigenvalues", "Linear Algebra"]


def test_rrf_fuses_query_views_once():
    a = SimpleNamespace(id=1)
    b = SimpleNamespace(id=2)
    c = SimpleNamespace(id=3)

    fused = rrf_fuse_many([
        ([(a, 0.9), (b, 0.8)], [(b, 4.0)]),
        ([(c, 0.95), (b, 0.7)], [(a, 3.0)]),
    ])

    ids = [item.id for item, _ in fused]
    assert set(ids) == {1, 2, 3}
    assert len(fused) == 3
