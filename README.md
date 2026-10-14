# Personal Learning Knowledge Agent v0.2.0

A self-organizing RAG + curriculum system for studying from lectures, transcripts, PDFs, books and other learning sources.

## What is new in v0.2

- Gemini 3.8 Flash for reasoning/synthesis with Gemini 3.5 Flash-Lite for high-volume background work and fallback.
- Manual transcript, SRT/VTT, Python-dict and JSON paste support.
- Optional source metadata: domain, topic, description and arbitrary JSON/Python-dict metadata.
- Structure-aware chunking and context-enriched indexing.
- Batched embeddings with `gemini-embedding-2`.
- Semantic groups organized by one decision per ~40 excerpts: groups are named from the content, concepts are tagged per excerpt reusing existing names and aliases, and centroids are the fallback.
- Hybrid retrieval: dense + BM25 + RRF + Gemini 3.5 Flash-Lite reranking.
- Context-window expansion for lecture/transcript continuity.
- Persistent jobs, checkpoints, retries and crash-safe resume.
- Course syllabus generation and copy-paste-ready study packs.
- Manual (paste-back) mode: without an API key, every model decision becomes a prompt you answer with any external AI; the paused job then finishes on its own.
- Near-duplicate merging: text repeated across sources in a course is stored once.
- Saved study outline with numbered topics and per-topic status, plus a guided session that feeds an external AI tutor one topic at a time.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# optional: add GEMINI_API_KEY (without it you paste requests into your own AI)
uvicorn app.main:app --reload
```

Open http://localhost:8000. The student page is served by the same process; the API stays at the same address (interactive docs at `/docs`).

## The student page

The page walks a student through three steps:

1. **Add material**: drop files, paste a transcript or notes (subtitle and JSON formats are detected automatically), or add a YouTube link. Each item shows a plain status: *Waiting in line*, *Reading your material*, *Checking for repeated content*, *Sorting it into topics*, *Needs your help*, *Ready* (with how many passages were kept and how many repeats were merged) or *Problem* with a **Try again** button.
2. **Study plan**: create the plan, then skip topics you already know or bring them back. When new material arrives later, the page suggests updating the plan; progress is kept.
3. **Study**: copy the first message into ChatGPT, Gemini or Claude, paste back the `START: …; SKIP: …` line the AI gives you, then copy one topic (or part) at a time and press **Next** when you're done with it.

Whenever the AI's help is needed (manual mode), a **Needs your help** card appears where it matters: copy the request, open your AI, paste its whole reply back. Model names, chunk ids and other internals are not shown.

The page is plain HTML, CSS and one JavaScript module in `frontend/` (no build step, no CDN, works offline), with light and dark themes and a phone-sized layout.

## AI modes

`AI_MODE` in `.env` picks who makes the model decisions:

| Mode | Decisions (organizing, outline) | Embeddings |
|---|---|---|
| `auto` (default) | Gemini when `GEMINI_API_KEY` is set, otherwise `manual` | as in the chosen mode |
| `gemini` | Gemini API | Gemini |
| `manual` | You, by pasting prompts into ChatGPT / Gemini / Claude | Gemini if a key is set, else local hashing |
| `local` | Offline fallbacks only (groups named after the source) | local hashing |

In manual mode a job that needs a decision shows status `waiting`. A **Needs your help** card appears on the page (or use `GET /decisions`), copy the prompt into your AI, paste its full reply back, and the job resumes from where it stopped. Replies wrapped in prose or code fences are accepted; a reply that is not valid JSON is rejected so you can paste it again.

## Studying with an external AI

1. **Add material.** Each source is split into pieces, repeated text is merged into the earlier copy, and the remaining pieces are organized into groups and concepts.
2. **Generate the study outline** (`POST /syllabus`). It is saved with numbered topics; every piece of the course belongs to exactly one topic. Regenerating keeps the status of topics whose title did not change.
3. **Kickoff** (`GET /tutor/kickoff`). Paste this into a new chat. It gives the AI the outline and tells it to ask you where to start and what to skip, and to answer with a line like `START: 3; SKIP: 5, 7`.
4. **Start** (`POST /tutor/start`). Paste that line back (or pick the start topic and skips yourself). Earlier and listed topics are marked `skipped` (topics already skipped in the plan stay skipped), and you get the first message to paste.
5. **Next** (`POST /tutor/next`). Each call returns the next topic, or the next part of a long topic, as verbatim excerpts in reading order, and records it as sent. Topics move through `pending → in_progress → done`; you can set any topic back to `pending` or `skipped` (`POST /syllabus/topics/{id}/status`), and `GET /tutor/topics/{id}/parts/{n}` re-sends a part without changing progress.

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
