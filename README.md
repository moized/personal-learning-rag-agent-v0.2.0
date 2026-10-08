# Personal Learning Knowledge System v0.4.0

LearnOS is a persistent, self-organizing learning system for lectures, transcripts, PDFs, books and other sources. It combines durable ingestion, hybrid retrieval, a knowledge/prerequisite graph, adaptive context construction and frontier-model reasoning.

## What is in v0.4

- Incremental source ingestion: add one source now and another later; multiple files are supported but never required.
- Crash-safe jobs with checkpoints, retries and compare-and-set queue claiming.
- Hierarchical parent/leaf chunks with deterministic document sequence links.
- Context-enriched batched embeddings and semantic grouping.
- Concept graph with prerequisite, related and contradiction relations.
- Adaptive multi-query retrieval: dense + BM25 → one RRF fusion → graph expansion → one rerank → parent/neighbor context.
- Course-aware routing with a global fallback when coverage is low.
- Study-run traces for debugging and regression analysis.
- Syllabus and study-pack generation.
- Lightweight Vite/React source-first learning workspace.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add GEMINI_API_KEY
uvicorn app.main:app --reload
```

The legacy Streamlit UI can still be run in a second shell:

```bash
streamlit run frontend/app.py
```

The new client is under web/ and remains intentionally lightweight.

## Model configuration

Defaults:

```text
reasoning = gemini-3.8-flash
fast       = gemini-3.5-flash-lite
fallback   = gemini-3.5-flash-lite
embedding  = gemini-embedding-2
```

Model selection is task-based. High-value synthesis uses the reasoning model while ingestion/routing/extraction uses the fast path.

The system also persists progress so temporary 429/5xx failures can be retried without rebuilding completed work.

## Manual transcript

The UI accepts:

1. plain transcript
2. SRT
3. VTT
4. Python dictionary literal
5. JSON

Python dictionary input is parsed with ast.literal_eval; arbitrary code is never executed.

## Knowledge graph

A source is represented as:

```
Course
 ↓
Source
 ↓
Document
 ↓
Parent windows
 ↓
Leaf chunks
 ↓
Concepts / Evidence
```

Concept prerequisites are directional:

```
Prerequisite → Dependent
```

At study time the system can backtrack from a target concept into its prerequisite evidence instead of relying only on text similarity.

## Evaluation

The repository includes lightweight retrieval metrics and an eval-set format. Start with 20–50 real questions before scaling the corpus.

See:

- docs/system_design.md
- docs/architecture_advanced.md
- docs/model_routing.md
- docs/evaluation.md
- docs/roadmap.md
- evals/README.md
