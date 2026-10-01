# Personal Learning Knowledge Agent v0.2.0

A self-organizing RAG + curriculum system for studying from lectures, transcripts, PDFs, books and other learning sources.

## What is new in v0.2

- Gemini 3.8 Flash for reasoning/synthesis with Gemini 3.5 Flash-Lite for high-volume background work and fallback.
- Manual transcript, SRT/VTT, Python-dict and JSON paste support.
- Optional source metadata: domain, topic, description and arbitrary JSON/Python-dict metadata.
- Structure-aware chunking and context-enriched indexing.
- Batched embeddings with `gemini-embedding-2`.
- Semantic groups with centroid routing and LLM validation only when ambiguous.
- Group-level concept extraction instead of one LLM call per chunk.
- Hybrid retrieval: dense + BM25 + RRF + Gemini 3.5 Flash-Lite reranking.
- Context-window expansion for lecture/transcript continuity.
- Persistent jobs, checkpoints, retries and crash-safe resume.
- Course syllabus generation and copy-paste-ready study packs.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add GEMINI_API_KEY
uvicorn app.main:app --reload
```

In a second shell:

```bash
streamlit run frontend/app.py
```

## Model configuration

Defaults:

```text
reasoning = gemini-3.8-flash
fast       = gemini-3.5-flash-lite
fallback   = gemini-3.5-flash-lite
embedding  = gemini-embedding-2
```

The free tier still has rate limits. The agent therefore batches embeddings and persists progress after each durable unit.

## Manual transcript

The UI accepts:

1. plain transcript
2. SRT
3. VTT
4. Python dictionary literal
5. JSON

Python-dict example:

```python
{
    "name": "Lecture 4",
    "domain": "AI / Information Retrieval",
    "topic": "Hybrid Retrieval",
    "description": "Dense + lexical retrieval and reranking.",
    "topics": ["BM25", "dense retrieval", "RRF", "reranking"],
    "transcript": "..."
}
```

The Python dictionary is parsed with `ast.literal_eval`; arbitrary code is never executed.

## Architecture

See:

- `docs/architecture_advanced.md`
- `docs/model_routing.md`
- `docs/evaluation.md`
- `docs/system_design.md`
- `docs/roadmap.md`
