# LearnOS Roadmap

## v0.4 — Adaptive Knowledge Architecture (current)

- hierarchical parent/leaf chunk structure
- deterministic next-chunk links
- concept relationship graph
- prerequisite traversal during retrieval
- adaptive query planning
- multi-query retrieval with one fusion and one rerank stage
- parent + neighbour context expansion
- durable graph workflow stage
- compare-and-set queue claiming
- study-run traces
- retrieval evaluation helpers and golden-set format

## v0.5 — Retrieval at scale

- Qdrant dense + sparse backend
- persistent lexical index
- embedding cache and embedding versioning
- incremental re-indexing by affected source/section
- better PDF section / heading extraction
- retrieval benchmark dashboard
- token / latency / cost ledger

## v0.6 — Learner intelligence

- learner event model
- concept mastery state
- misconception tracking
- adaptive curriculum
- quiz generation and grading
- spaced review

## v0.7 — Research agent

- web discovery tools
- source quality validation
- date and freshness filtering
- evidence comparison
- current vs historical topic detection
- research artifacts stored outside the prompt
- trusted-source syllabus refresh

## v0.8 — Multimodal learning

- local audio/video ingestion
- timestamped speech-to-text
- slide and figure extraction
- equation-aware evidence
- multimodal embeddings
- frame / audio / page citations

## Later

- explicit work-item leases for multiple workers
- optional MCP tool server
- advanced late-interaction reranking
- graph communities / global search
- model-provider adapters for multiple frontier vendors
