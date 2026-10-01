# Implementation status — v0.3.0

## Verified locally in development

- Persistent SQLite source/job/chunk state.
- Multiple independent sources can be queued without waiting for earlier sources.
- Running/retry jobs are reclaimable after restart.
- Completed chunk work is not recomputed.
- PDF/TXT/MD/DOCX/SRT/VTT/manual transcript ingestion.
- Python-dict/JSON paste is parsed safely with ast.literal_eval fallback.
- YouTube video and playlist ingestion path with video-level progress.
- Optional source metadata and source profiling.
- Batched embeddings and hybrid dense + BM25 + RRF retrieval.
- Optional Gemini LLM reranking and source-grounded synthesis.
- Course syllabus generation and copy-paste study packs.
- Vite + React frontend with incremental Add workflow and automatic queue polling.

## Known production-hardening work

- Qdrant is still a planned scale backend rather than the default local backend.
- Queue claiming is intended for one local worker; multi-worker atomic claiming is a later step.
- Arbitrary local video-file speech-to-text is not yet built in.
- Concept deduplication and syllabus quality need a real evaluation set before trusting large corpora.
- Web freshness/research agent is a later layer.
- The agent is a durable workflow controller, not self-modifying code.
