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

## Duplicate merging

Before embedding, each new chunk is compared with the earlier chunks of the same course using word 3-gram shingles, ignoring the overlap the chunker copies from the previous chunk. When at least 90% (`DUPLICATE_THRESHOLD`) of a chunk's text already exists in one earlier chunk, it is marked as a duplicate of that chunk. Duplicates are not embedded, organized, retrieved or sent to the tutor; the canonical excerpt notes the other sources it appeared in. Paraphrases are not merged; they end up in the same group and concepts instead.

## Efficient grouping

Chunks are organized in batches of about 40 (`ORGANIZE_BATCH_SIZE`) with one model decision per batch. The prompt lists the existing groups (all of them, or the 40 closest to the batch by centroid) with their concepts. The model puts every excerpt in an existing group or a new group named after its content, and lists the concepts each excerpt teaches, reusing existing concept names. Group and concept names are matched case-, punctuation- and plural-insensitively, and concept aliases resolve to the same concept. Excerpts the answer leaves out fall back to the nearest group by centroid, then to their neighbour's group.

Concept tagging happens in the same decision, so concepts link to the exact chunks that teach them.

## Decisions and manual mode

Model calls go through a provider. In `manual` mode the provider raises `DecisionPending` instead of calling an API. The worker records a `Decision` (prompt, expected JSON shape, linked job) and marks the job `waiting`. When the user pastes the external AI's reply, it is validated and stored, the job is re-queued, and the replayed step reads the stored answer. Organize decisions use a stable key per chunk range, so an answer still applies when the list of groups changed while it was pending.

## Curriculum

The syllabus decision orders topics so prerequisites come first, says which groups and concepts each topic covers, and reports concept names that mean the same thing; those are merged in the database (evidence moved, names kept as aliases). The outline is saved as numbered `SyllabusTopic` rows with a status (`pending`, `in_progress`, `done`, `skipped`). User outlines are optional planning preferences and are not treated as evidence.

## Study

Every canonical chunk of a course belongs to exactly one topic: the topic claiming most of its concepts, otherwise the topic claiming its group, otherwise an automatic "Additional material" topic. The tutor flow sends a kickoff message (outline and "ask me where to start"), applies the user's `START`/`SKIP` choice, and then returns one topic at a time as verbatim excerpts in reading order, split into parts of about 15,000 characters (`STUDY_PART_CHARS`).

For a free-form topic, the retrieval engine can still build a one-off study pack with cited chunks for a separate ChatGPT/Gemini session.

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
