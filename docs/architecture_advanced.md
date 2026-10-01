# Advanced RAG Architecture — v0.2

## Goal

This project is intentionally more sophisticated than a basic `embed -> similarity -> prompt` RAG bot, while remaining runnable on a normal CPU machine and compatible with free Gemini API usage.

## 1. Architecture

```text
                       USER
                         |
       +-----------------+------------------+
       |                 |                  |
    PDF/DOCX          Transcript       YouTube URL
       |                 |                  |
       +-----------------+------------------+
                         |
                    Source Adapter
                         |
                 Persistent Job Queue
                         |
          +--------------+--------------+
          |              |              |
        Parse          Chunk          Metadata
          |              |              |
          +--------------+--------------+
                         |
               Source / Document Profile
                         |
               Context-enriched chunks
                         |
               Batched Gemini Embedding
                         |
               Semantic Group Routing
               /                    \\
        centroid fast-path      LLM validation
               \                    /
                +------------------+
                         |
                    Knowledge Groups
                         |
                Group-level Concepts
                         |
                   Course Curriculum
                         |
        +----------------+------------------+
        |                                   |
      Query                              Study topic
        |                                   |
  Query planning                            |
        |                                   |
  Dense retrieval + BM25                    |
        |                                   |
       RRF fusion                            |
        |                                   |
   Gemini Lite reranker                      |
        |                                   |
 Context-window expansion                    |
        |                                   |
 Gemini 3.8 synthesis                        |
        |                                   |
 Answer / Study Pack / Quiz                  |
```

## 2. Model routing

### Gemini 3.8 Flash

Use for:

- syllabus design
- study-pack synthesis
- complex multi-step planning
- final answer generation
- future agent planning/tool orchestration

Default thinking level: `medium`.

### Gemini 3.5 Flash-Lite

Use for:

- source profiling
- routing/classification
- query expansion
- reranking
- concept extraction
- contextualization
- recovery fallback when the primary model fails

Default thinking level: `minimal`.

The code routes by **task**, not by one global model setting. This prevents a large source from paying the most expensive reasoning path for every tiny extraction step.

Both models currently have a free API tier according to Google's pricing documentation. Limits are still rate-limited, so the workflow also uses batching, retry/backoff and checkpoints.

## 3. Ingestion

Supported source types:

- PDF
- TXT / Markdown
- DOCX
- SRT / VTT
- manually pasted transcript
- Python dictionary literal or JSON pasted into the manual-source box
- YouTube video / playlist

The metadata fields are optional. They improve routing, but content remains the primary signal.

## 4. Chunking

The chunker is structure-aware rather than a blind fixed-width splitter:

1. paragraph boundaries
2. sentence boundaries for long paragraphs
3. bounded overlap
4. source/document/section metadata preserved

For lectures, retrieval can additionally expand to adjacent chunks from the same document. This preserves the local narrative around a selected passage.

## 5. Context-aware indexing

Every chunk receives a deterministic contextual prefix such as:

```text
Source | Lecture title | Section | Domain | Topic hint
```

An optional LLM contextualization mode can generate a short chunk-specific context sentence for small sources. It is disabled by default for large corpora because it multiplies API calls.

This is inspired by the contextual retrieval pattern in which chunk-specific context improves lexical and semantic retrieval.

## 6. Retrieval

The runtime retrieval pipeline is:

```text
query
  -> optional query expansion
  -> dense candidate retrieval
  -> BM25 candidate retrieval
  -> reciprocal rank fusion (RRF)
  -> Gemini Lite reranking
  -> local context-window expansion
  -> final context
```

Dense retrieval is strong for semantic similarity. BM25 is strong for exact technical terminology, identifiers and phrases. Rank fusion combines both signals before the expensive reranking step.

## 7. Group-aware retrieval

Group centroids are used to restrict the first retrieval pass:

```text
primary group
   + closest related groups
   + global fallback when coverage is too low
```

A bad classification therefore reduces noise without making relevant information permanently inaccessible.

## 8. Semantic grouping

New chunks are not compared against every existing chunk.

Instead:

```text
chunk embedding
   -> nearest group centroids
   -> strong-margin deterministic assignment
   -> LLM validation only when ambiguous
```

This keeps grouping cheap as the number of chunks grows.

## 9. Concept extraction

Concept extraction is **group-level**, not one LLM call per chunk.

That change is deliberate:

```text
1000 chunks -> potentially 1000 extraction calls   [bad]

1000 chunks -> 15 semantic groups -> ~15 extraction calls [much better]
```

Representative evidence snippets are used to build the concept index and a small number of high-signal evidence links are retained.

## 10. Durable execution

The agent runtime is deliberately separate from the model.

The model decides or generates content; SQLite stores the workflow state.

```text
job status
stage
progress
checkpoint
retry count
error
plan
```

On restart, `running` jobs are reclaimed and incomplete chunks/groups continue without replaying completed work.

This is a local durable workflow, not a self-modifying autonomous program.

## 11. Why we do not force LangGraph into v0.2

A graph orchestration framework can be useful later, but the current workflow already requires durable local state, and SQLite gives a transparent source of truth for a free/local project. Introducing another orchestration runtime now would make debugging harder without solving the immediate ingestion/retrieval problem.

A future adapter can expose the same stage machine through a graph runtime without changing the domain model.

## 12. Qdrant path

The project keeps its storage/retrieval core dependency-light for local testing. A Qdrant backend should be added when the corpus becomes large enough that in-process dense scanning is the bottleneck.

The planned Qdrant architecture is:

```text
Qdrant dense vector
+ sparse/BM25 retrieval
+ reranking
```

This mirrors the modern hybrid pattern demonstrated in Qdrant's documentation.

## 13. Evaluation before scaling

Before importing 25+ hours of material, test a small golden set:

```text
3-5 lectures
1 PDF
20-50 manually written questions
```

Measure:

- Recall@k
- MRR / ranking quality
- citation/source accuracy
- duplicate concept rate
- grouping accuracy
- syllabus coherence
- resume correctness
- API calls per source
- processing time per 100 chunks

Only after the small evaluation passes should the corpus be expanded.

## 14. Current limitations

- YouTube transcript ingestion still depends on the platform exposing a usable transcript.
- Arbitrary video files do not yet have a built-in speech-to-text path.
- Dense storage is currently in SQLite JSON for zero-service local operation.
- LLM reranking is a practical free-tier alternative to a dedicated late-interaction reranker, not a replacement for a high-scale specialized reranker.
- Web research/source validation remains a future layer.
