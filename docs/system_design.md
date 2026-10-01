# Personal Learning Knowledge Agent — System Design v0.2

## Vision

A persistent personal learning system that turns heterogeneous learning sources into an organized knowledge base, then turns that knowledge into a course syllabus and evidence-grounded study material.

This is not a basic `vector DB + chatbot` design. The v0.2 retrieval architecture is:

```text
context-enriched chunks
      ↓
Gemini Embedding 2
      ↓
semantic groups
      ↓
course-aware query routing
      ↓
Dense retrieval + BM25
      ↓
RRF fusion
      ↓
Gemini 3.5 Flash-Lite reranking
      ↓
context-window expansion
      ↓
Gemini 3.8 Flash synthesis
```

## Sources

- YouTube video / playlist
- PDF
- TXT / Markdown
- DOCX
- SRT / VTT
- pasted transcript
- pasted JSON
- pasted Python dictionary literal

## Metadata

Optional fields include course, domain, topic, description, lecturer, author, week, level and arbitrary metadata.

Hints are routing signals, not truth. The source content can disagree with the hints and the semantic index can route the material elsewhere.

## Durable workflow

Each source creates a persistent job. The worker stores:

- status
- current stage
- progress
- checkpoint
- retry count
- error
- execution plan

Every chunk stores independent completion state for embedding/classification/concept indexing. Completed work is skipped during resume.

## Organization

The system maintains three useful levels:

```text
Course
  ↓
Semantic Group
  ↓
Concept / Evidence
```

A course is a user learning container. Groups are semantic areas shared across many sources. Concepts are canonical learning units with evidence links.

## Efficient grouping

New chunks are compared to group centroids rather than every chunk. Strong-margin matches are assigned without an LLM call. Only ambiguous candidates use LLM validation.

Concept extraction is group-level so the system does not perform one expensive concept extraction call per chunk.

## Curriculum

The syllabus engine merges repeated concepts across sources and asks Gemini 3.8 Flash to order prerequisites before dependent topics. User outlines are optional planning preferences and are not treated as evidence.

## Study

For a selected topic, the retrieval engine finds evidence from the selected course, builds a study pack, cites the source chunks, and produces copy-paste-ready text for a separate ChatGPT/Gemini tutoring session.

## Modern architecture decisions

### Adopted

- structure-aware chunking
- context-enriched indexing
- dense + lexical hybrid retrieval
- Reciprocal Rank Fusion
- reranking only after candidate reduction
- query planning
- context-window expansion
- persistent checkpoints
- task-aware model routing
- structured model outputs

### Intentionally optional

- LLM-generated contextualization for every chunk: useful but API-expensive; enabled only for small sources if configured
- dedicated Qdrant/ANN vector infrastructure: useful as corpus size grows; the current SQLite/NumPy path keeps v0.2 immediately runnable and testable
- specialized ColBERT/late-interaction reranking: excellent for larger deployments, but Gemini Lite reranking is easier to run with the current free-first constraints

## Model routing

```text
Gemini 3.5 Flash-Lite
  → classification / routing / query planning / reranking / concept extraction

Gemini 3.8 Flash
  → syllabus / study synthesis / difficult reasoning / future agent planning

Gemini Embedding 2
  → semantic embeddings
```

## Primary design principle

Use the strongest technique that materially improves this project's learning workflow, but avoid adding infrastructure that creates cost or complexity without a measurable retrieval/learning benefit.
