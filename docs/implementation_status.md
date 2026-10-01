# Implementation status — v0.1.1

## Verified locally

- FastAPI app imports and `/health` path remains available.
- Persistent SQLite source/job/chunk state.
- Multiple sources can be queued independently.
- Running/retry jobs are reclaimable after restart.
- Per-chunk states prevent completed chunks from being recomputed.
- PDF/TXT/MD/DOCX ingestion.
- Manual transcript ingestion through text or structured JSON.
- SRT/VTT cleanup.
- YouTube video/playlist path is implemented.
- Optional course, topic, description, and metadata fields.
- Course-scoped syllabus generation.
- Group-aware study retrieval.
- Copy-paste study-pack endpoint for external AI tutors.
- Local fallback provider for offline unit tests.

## Not claimed as production-complete

- Qdrant is not yet wired into the main retrieval path.
- Multi-worker atomic queue claiming is not implemented.
- Arbitrary video-file speech-to-text fallback is not implemented.
- Concept deduplication needs benchmark data before large-scale trust.
- Web research/currentness agent is a future layer.
- The planner is a durable deterministic agent controller, not self-modifying code.
