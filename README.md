# Personal Learning Knowledge Agent v0.3.0

A self-organizing RAG + curriculum system for studying from lectures, transcripts, PDFs, books and other learning sources.

## What is in v0.3

- Persistent source/job workflow with checkpoints and resume.
- PDF, DOCX, TXT/Markdown, SRT/VTT, pasted transcript, Python dictionary/JSON and YouTube ingestion.
- Context-aware chunking, batched Gemini embeddings, semantic grouping and group-level concepts.
- Hybrid dense + BM25 + RRF + LLM reranking with local context expansion.
- Course-level syllabus generation and copy-paste study packs.
- Lightweight Vite/React learning workspace for incremental source addition.

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
