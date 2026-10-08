from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for an adaptive, durable learning system."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Personal Learning Knowledge Agent"
    database_url: str = "sqlite:///./data/app.db"
    upload_dir: str = "./data/uploads"

    gemini_api_key: str | None = None
    gemini_reasoning_model: str = "gemini-3.8-flash"
    gemini_fast_model: str = "gemini-3.5-flash-lite"
    gemini_fallback_model: str = "gemini-3.5-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-2"
    embedding_dim: int = 768
    embedding_batch_size: int = 16
    gemini_reasoning_level: str = "medium"
    gemini_fast_thinking_level: str = "minimal"

    chunk_size: int = 1800
    chunk_overlap: int = 220
    parent_chunk_window: int = 3
    contextual_indexing: bool = False
    contextual_indexing_max_chunks: int = 80
    group_similarity_threshold: float = 0.78
    group_candidates: int = 8
    group_llm_validation: bool = True
    concept_extraction: bool = True
    concept_relationships: bool = True
    relationship_min_confidence: float = 0.72

    dense_candidate_k: int = 24
    lexical_candidate_k: int = 24
    fused_candidate_k: int = 30
    final_context_k: int = 8
    retrieval_group_limit: int = 4
    prerequisite_max_depth: int = 2
    prerequisite_evidence_k: int = 8
    use_llm_query_planner: bool = True
    use_llm_reranker: bool = True

    worker_count: int = 1
    poll_seconds: float = 2.0
    retry_max: int = 4

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)


settings = Settings()
settings.upload_path.mkdir(parents=True, exist_ok=True)
Path("./data").mkdir(exist_ok=True)
