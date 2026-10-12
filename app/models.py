from datetime import datetime, UTC
from sqlalchemy import String, Text, Integer, Float, ForeignKey, UniqueConstraint, DateTime, Index, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column
from .db import Base


def utcnow():
    return datetime.now(UTC)


class Course(Base):
    __tablename__ = "courses"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(500), unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    domain: Mapped[str | None] = mapped_column(String(255), nullable=True)
    outline: Mapped[str | None] = mapped_column(Text, nullable=True)
    # "learnos" when LearnOS made the space while filing a source (removed once empty), "you" when the learner did.
    created_by: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class StudyProgress(Base):
    __tablename__ = "study_progress"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scope_key: Mapped[str] = mapped_column(String(80), index=True)
    topic_key: Mapped[str] = mapped_column(String(500))
    topic_label: Mapped[str] = mapped_column(String(500))
    # The study-plan topic this progress belongs to; survives a rebuild that rewords the title.
    node_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    completed: Mapped[bool] = mapped_column(default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    __table_args__ = (UniqueConstraint("scope_key", "topic_key", name="uq_study_progress_scope_topic"),)


class Source(Base):
    __tablename__ = "sources"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(500))
    source_type: Mapped[str] = mapped_column(String(50))
    uri: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    local_path: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    course_id: Mapped[int | None] = mapped_column(ForeignKey("courses.id"), nullable=True, index=True)
    assignment_mode: Mapped[str] = mapped_column(String(30), default="explicit_unassigned")
    assignment_status: Mapped[str] = mapped_column(String(30), default="unassigned")
    assignment_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    domain_hint: Mapped[str | None] = mapped_column(String(255), nullable=True)
    topic_hint: Mapped[str | None] = mapped_column(String(500), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="queued")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    profile_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # This source's own topic outline, reused by study plans until its content fingerprint changes.
    curriculum_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    curriculum_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Mean embedding of an even sample of its passages, and what it was computed from.
    centroid_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    centroid_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # The learner's answers to placement suggestions, e.g. {"kept_out_of": [3]}.
    placement_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class SourceSpace(Base):
    """A space a source is also shown in, besides the one it is filed in (Source.course_id)."""
    __tablename__ = "source_spaces"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    __table_args__ = (UniqueConstraint("source_id", "course_id", name="uq_source_space"),)


class Group(Base):
    __tablename__ = "groups"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    centroid_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), index=True)
    external_id: Mapped[str | None] = mapped_column(String(500), nullable=True)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    section: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Platform facts such as title, channel, upload date and duration, plus transcript origin.
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    parent_chunk_id: Mapped[int | None] = mapped_column(ForeignKey("chunks.id"), nullable=True, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    chunk_kind: Mapped[str] = mapped_column(String(20), default="leaf", index=True)
    text: Mapped[str] = mapped_column(Text)
    contextual_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    embedding_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_status: Mapped[str] = mapped_column(String(30), default="pending")
    classification_status: Mapped[str] = mapped_column(String(30), default="pending")
    cluster_status: Mapped[str] = mapped_column(String(30), default="pending")
    group_id: Mapped[int | None] = mapped_column(ForeignKey("groups.id"), nullable=True, index=True)
    group_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # The section (video, chapter, slides, heading) this leaf belongs to, rebuilt from the source's structure.
    section_id: Mapped[int | None] = mapped_column(ForeignKey("sections.id"), nullable=True, index=True)
    # Version of the concept extraction that has read this chunk's full text; 0 = not read yet.
    concept_pass: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk"),
        Index("ix_chunk_document_kind_index", "document_id", "chunk_kind", "chunk_index"),
    )


class Section(Base):
    """A titled part of a source taken from its own structure: a video, a PDF chapter, slides, a heading."""
    __tablename__ = "sections"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), index=True)
    document_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id"), nullable=True, index=True)
    position: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str] = mapped_column(String(500))
    kind: Mapped[str] = mapped_column(String(20))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ChunkRelation(Base):
    __tablename__ = "chunk_relations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.id"), index=True)
    to_chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.id"), index=True)
    relation_type: Mapped[str] = mapped_column(String(50), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    __table_args__ = (
        UniqueConstraint("from_chunk_id", "to_chunk_id", "relation_type", name="uq_chunk_relation"),
    )


class Concept(Base):
    __tablename__ = "concepts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(255), unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    aliases_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    group_id: Mapped[int | None] = mapped_column(ForeignKey("groups.id"), nullable=True, index=True)


class ConceptRelation(Base):
    __tablename__ = "concept_relations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_concept_id: Mapped[int] = mapped_column(ForeignKey("concepts.id"), index=True)
    to_concept_id: Mapped[int] = mapped_column(ForeignKey("concepts.id"), index=True)
    relation_type: Mapped[str] = mapped_column(String(50), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    __table_args__ = (
        UniqueConstraint("from_concept_id", "to_concept_id", "relation_type", name="uq_concept_relation"),
    )


class Evidence(Base):
    __tablename__ = "evidence"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    concept_id: Mapped[int] = mapped_column(ForeignKey("concepts.id"), index=True)
    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.id"), index=True)
    relation: Mapped[str] = mapped_column(String(50), default="supports")
    score: Mapped[float] = mapped_column(Float, default=1.0)
    __table_args__ = (UniqueConstraint("concept_id", "chunk_id", name="uq_concept_chunk"),)


class AgentTrace(Base):
    __tablename__ = "agent_traces"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    parent_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(String(50), index=True)
    task: Mapped[str | None] = mapped_column(String(80), nullable=True)
    model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="ok")
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), index=True)
    task: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(50), default="ingest")
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    plan_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress_current: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    checkpoint: Mapped[str | None] = mapped_column(String(500), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class StudyPlan(Base):
    __tablename__ = "study_plans"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scope_key: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    revision: Mapped[str] = mapped_column(String(64))
    syllabus: Mapped[str] = mapped_column(Text)
    relationships_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class StudyPlanNode(Base):
    __tablename__ = "study_plan_nodes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plan_id: Mapped[int] = mapped_column(ForeignKey("study_plans.id", ondelete="CASCADE"), index=True)
    node_id: Mapped[str] = mapped_column(String(100))
    parent_node_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    node_type: Mapped[str] = mapped_column(String(20))
    path_id: Mapped[str] = mapped_column(String(100), default="main", index=True)
    title: Mapped[str] = mapped_column(String(500))
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    __table_args__ = (
        UniqueConstraint("plan_id", "node_id", name="uq_plan_node"),
        CheckConstraint("node_type IN ('outline', 'study')", name="ck_study_plan_node_type"),
    )


class TopicDetail(Base):
    """A plan topic (or one of its parts) prepared for a study session of a given size.

    Plan topics are prepared on demand: the result is a study unit (a pack of source passages
    that fits the session) or a split into smaller parts, which are rows of their own. Rows are
    reused while what their topic was made from (the plan's content and the topic's place in it) is unchanged.
    """
    __tablename__ = "topic_details"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scope_key: Mapped[str] = mapped_column(String(80), index=True)
    node_id: Mapped[str] = mapped_column(String(100), index=True)
    parent_node_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    depth: Mapped[int] = mapped_column(Integer, default=0)
    position: Mapped[int] = mapped_column(Integer, default=0)  # order among its parent's parts
    budget_tokens: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(500))
    purpose: Mapped[str | None] = mapped_column(Text, nullable=True)
    # For parts: the passages assigned to them when their parent was split.
    chunk_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | unit | thin | split
    # The version of the plan topic it was prepared from (see units.Topic.version).
    plan_revision: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    __table_args__ = (UniqueConstraint("scope_key", "node_id", "budget_tokens", name="uq_topic_detail"),)


class SourceRelationship(Base):
    __tablename__ = "source_relationships"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_a_id: Mapped[int] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)
    source_b_id: Mapped[int] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)
    source_a_profile_hash: Mapped[str] = mapped_column(String(64))
    source_b_profile_hash: Mapped[str] = mapped_column(String(64))
    relation_type: Mapped[str] = mapped_column(String(40))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    rationale: Mapped[str] = mapped_column(Text)
    evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    __table_args__ = (
        UniqueConstraint("source_a_id", "source_b_id", name="uq_source_relationship_pair"),
        CheckConstraint("relation_type IN ('SAME_COURSE','SAME_LEARNING_PATH','COMPLEMENTARY_RESOURCE','SAME_DOMAIN_DIFFERENT_COURSE','RELATED_BUT_SEPARATE','UNRELATED')", name="ck_source_relationship_type"),
    )


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(500), default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, index=True)


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class MaintenanceJob(Base):
    __tablename__ = "maintenance_jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task: Mapped[str] = mapped_column(String(60))
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    progress_current: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    checkpoint: Mapped[str | None] = mapped_column(String(500), nullable=True)
    items_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    errors_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, index=True)


class LearnerEvent(Base):
    """Append-only log of everything the learner does; mastery is derived from it."""
    __tablename__ = "learner_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_type: Mapped[str] = mapped_column(String(40), index=True)
    concept_id: Mapped[int | None] = mapped_column(ForeignKey("concepts.id"), nullable=True, index=True)
    course_id: Mapped[int | None] = mapped_column(ForeignKey("courses.id"), nullable=True, index=True)
    quiz_item_id: Mapped[int | None] = mapped_column(ForeignKey("quiz_items.id"), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class ConceptMastery(Base):
    """Current learner state for one concept, including its SM-2 review schedule."""
    __tablename__ = "concept_mastery"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    concept_id: Mapped[int] = mapped_column(ForeignKey("concepts.id"), unique=True, index=True)
    mastery: Mapped[float] = mapped_column(Float, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    correct_count: Mapped[int] = mapped_column(Integer, default=0)
    ease: Mapped[float] = mapped_column(Float, default=2.5)
    interval_days: Mapped[float] = mapped_column(Float, default=0.0)
    repetitions: Mapped[int] = mapped_column(Integer, default=0)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    next_due_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)


class Misconception(Base):
    __tablename__ = "misconceptions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    concept_id: Mapped[int] = mapped_column(ForeignKey("concepts.id"), index=True)
    description: Mapped[str] = mapped_column(Text)
    occurrences: Mapped[int] = mapped_column(Integer, default=1)
    resolved: Mapped[bool] = mapped_column(default=False, index=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class QuizItem(Base):
    __tablename__ = "quiz_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    concept_id: Mapped[int] = mapped_column(ForeignKey("concepts.id"), index=True)
    course_id: Mapped[int | None] = mapped_column(ForeignKey("courses.id"), nullable=True, index=True)
    question: Mapped[str] = mapped_column(Text)
    reference_answer: Mapped[str] = mapped_column(Text)
    evidence_chunk_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
