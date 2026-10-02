# Personal Learning Knowledge System — System Design v0.4

## Vision

LearnOS is a persistent personal learning system built to stay useful as frontier models improve. Models provide reasoning and language generation; the durable value of the system lives in its knowledge graph, evidence, learner state, retrieval engine, workflow state and evaluation suite.

The system is intentionally not a "vector database + chatbot".

## System shape

\`\`\`
User
 ↓
Learning Agent / Orchestrator
 ├─ retrieval tools
 ├─ knowledge-graph tools
 ├─ curriculum tools
 ├─ learner-state tools
 └─ optional web / research tools
 ↓
Context Engine
 ├─ dense retrieval
 ├─ lexical retrieval
 ├─ query decomposition
 ├─ graph traversal
 ├─ parent / neighbour expansion
 └─ adaptive context budgeting
 ↓
Strong reasoning model
 ↓
Evidence-grounded artifact
 ├─ answer
 ├─ study pack
 ├─ quiz
 └─ plan
 ↓
Verification + trace + persistent state
\`\`\`

## Source ingestion

Supported inputs remain incremental and independent:

- PDF
- DOCX
- TXT / Markdown
- SRT / VTT
- pasted transcript
- JSON / Python dictionary literal
- YouTube video / playlist

A user may add one source now and another later. Multiple files are supported but never required. Every source creates its own durable job.

Optional metadata remains a weak routing signal. Content is authoritative.

## Hierarchical content model

The source tree is represented as:

\`\`\`
Course
  ↓
Source
  ↓
Document
  ↓
Parent chunk window
  ↓
Leaf chunks
\`\`\`

Leaf chunks are the retrieval unit. Parent windows preserve several adjacent leaf chunks so the synthesis model can recover local context without forcing every retrieval call to use large chunks.

Each leaf has:

- document position
- parent_chunk_id
- contextualized text
- summary
- embedding
- semantic group
- processing states

Deterministic sequence relations preserve narrative order.

## Knowledge graph

The graph has two complementary layers.

### Content relations

\`\`\`
leaf A ──next──→ leaf B
leaf B ──next──→ leaf C
\`\`\`

### Concept relations

\`\`\`
Linear Algebra ──prerequisite──→ Eigenvalues
Eigenvalues ──prerequisite──→ PCA
Concept A ──related──→ Concept B
Concept C ──contradicts──→ Concept D
\`\`\`

Prerequisite edges are directed from prerequisite to dependent concept.

This enables backward traversal:

\`\`\`
PCA
 ↓ prerequisite
Eigenvalues
 ↓ prerequisite
Linear Algebra
\`\`\`

The graph is evidence-backed. A relation stores confidence and may later store supporting source references.

## Grouping and concepts

New leaf embeddings are compared with semantic group centroids. Strong-margin matches bypass the LLM; ambiguous candidates are validated with the fast model.

Concept extraction is group-level rather than per chunk to reduce API calls.

After concept extraction, relationship extraction identifies high-confidence prerequisite, related and contradiction edges inside the relevant concept set.

As the system evolves, concept identity should move beyond exact string equality so the same concept can accumulate evidence from multiple courses.

## Adaptive retrieval

Runtime retrieval is no longer fixed top-k RAG.

\`\`\`
Question
 ↓
Fast query planner
 ├─ 1–3 search views
 ├─ target concepts
 ├─ prerequisite need / depth
 ├─ source-comparison flag
 └─ complexity
 ↓
Group routing
 ↓
Parallel candidate generation
 ├─ dense
 └─ BM25 / lexical
 ↓
One RRF fusion
 ↓
Knowledge-graph prerequisite expansion when requested
 ↓
One reranking pass
 ↓
Parent + neighbour context expansion
 ↓
Adaptive final context
\`\`\`

The important design choice is that multiple query variants share one candidate pool and one rerank step. We do not pay for a separate reranker invocation per query variant.

A low-coverage result can escape the group restriction once via global fallback.

## Context engineering

The system treats context as a managed resource rather than a fixed prompt payload.

For small course scopes, long-context reasoning can replace unnecessary retrieval. For medium and large corpora, retrieval narrows the context before reasoning.

Future context policies can consider:

- question complexity
- evidence quality
- redundancy
- prerequisite coverage
- source diversity
- token budget
- cacheability

## Agent architecture

The model does not own the durable workflow.

\`\`\`
Agent
  ↓ observe state
  ↓ choose next action
  ↓ call tool
  ↓ inspect result
  ↓ continue / recover / stop
\`\`\`

The local runtime keeps durable source and job state in SQLite. This keeps the current system transparent and easy to debug while leaving room for a richer graph runtime later.

Tool boundaries are explicit so stronger frontier models can be adopted without rewriting the domain model.

Planned internal tools include:

- search_knowledge
- search_concepts
- traverse_prerequisites
- get_parent_context
- compare_sources
- get_syllabus
- get_learner_state
- record_learning_event
- generate_quiz
- verify_answer
- search_web

## Durable execution

Each source job persists:

- status
- plan
- stage
- progress
- checkpoint
- retry count
- error

Chunk-level processing is idempotent. Existing embeddings and classifications are reused.

Queue claiming uses compare-and-set semantics so the same queued job is not normally claimed by two workers concurrently.

The next scale step is explicit work items / leases for multi-worker execution.

## Agent traces

Study runs persist a compact trace containing:

- run id
- model/task
- query plan
- selected groups
- target concepts
- retrieved chunk ids
- latency
- result counts

This gives us a local audit/debug surface and creates the foundation for regression analysis.

## Model routing

Default routing remains task-based:

\`\`\`
Fast model
 → profiling / routing / extraction / relationships / reranking

Reasoning model
 → syllabus / study synthesis / difficult planning

Embedding model
 → dense representations
\`\`\`

The architecture intentionally avoids hard-coding one vendor into the system domain. The provider layer should remain replaceable.

## Retrieval backend evolution

Current local mode:

\`\`\`
SQLite
 → metadata + workflow + graph + embeddings
\`\`\`

Scale-up mode:

\`\`\`
SQLite
 → metadata + workflow + graph

Qdrant
 → dense + sparse + future multivector retrieval
\`\`\`

Qdrant should be introduced when corpus size or latency makes in-process scanning the bottleneck, not simply because a vector database is fashionable.

## Evaluation

A small golden set is part of the product architecture.

Start with 20–50 real questions and measure:

- Recall@5 / Recall@10
- MRR
- citation correctness
- prerequisite traversal usefulness
- group accuracy
- duplicate concept rate
- resume correctness
- duplicate-ingestion safety
- processing time
- model/API call counts

Every significant model or retrieval change should run the regression suite.

## Explicit non-goals

- self-modifying source code
- autonomous internet browsing for every question
- multi-agent orchestration for trivial questions
- mandatory cloud infrastructure
- forcing the user to upload all materials at once

## Design principle

The model may change tomorrow. The knowledge, evidence, graph, state, tools and evaluations should continue working.
