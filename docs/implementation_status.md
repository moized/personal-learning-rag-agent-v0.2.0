# Implementation status — v0.4.0

## Implemented

- Persistent SQLite source/job/chunk state.
- Incremental source ingestion with independent jobs.
- Crash/retry recovery and compare-and-set queue claiming.
- PDF/TXT/MD/DOCX/SRT/VTT/manual transcript ingestion.
- Safe JSON/Python-dict parsing with ast.literal_eval fallback.
- YouTube video and playlist ingestion with video-level checkpoints.
- Hierarchical parent/leaf chunk windows and deterministic next-chunk relations.
- Context-enriched embeddings, semantic grouping and group-level concept extraction.
- Cross-group concept relationship extraction with source-chunk provenance for prerequisite, parallel, related and contradiction edges.
- Adaptive multi-query dense + BM25 retrieval with one RRF fusion and one optional LLM rerank.
- Prerequisite graph traversal can inject supporting evidence before reranking.
- Parent/neighbor context expansion after reranking.
- Low-coverage global fallback.
- Study-run trace persistence and inspection endpoint.
- Course knowledge-graph endpoint.
- Syllabus generation, study packs and Vite/React incremental source UI.
- Initial retrieval evaluation helpers and 20–50-question eval format.

## v0.6 learner intelligence (backend, unreleased)

- Append-only learner event log; `/study` logs a `question` event per resolved concept.
- Per-concept mastery with forgetting-curve decay and SM-2 spaced-review scheduling.
- Evidence-grounded quiz generation and LLM grading with an offline keyword-overlap fallback.
- Misconception tracking, fed back into tutor answers and quiz generation.
- Adaptive "what next": due reviews, weak concepts, then concepts whose prerequisites are mastered.
- Endpoints: `POST /quiz/generate`, `POST /quiz/{id}/answer`, `POST /learner/self-rating`,
  `GET /learner/{mastery,due,next,misconceptions}`, `POST /learner/misconceptions/{id}/resolve`.
- Not yet: web UI for review/quiz/progress, adaptive reordering of the persisted study plan.

## Intentionally not the default yet

- Qdrant / sparse ANN infrastructure.
- Dedicated late-interaction reranker.
- Full learner mastery model.
- Web research worker.
- Multimodal media indexing.
- Multi-worker work-item leases.

These are staged behind measurements rather than added solely for architectural fashion.

## Validation note

The environment used to edit the repository cannot reliably clone GitHub over outbound DNS, so local execution of the full application has not been claimed from this environment. GitHub Actions remains the authoritative CI path for the branch.

## Design direction

The durable system assets are knowledge, evidence, graph, state and evaluation. The model provider is replaceable.
