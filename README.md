# LearnOS

LearnOS turns your own study material (YouTube videos and playlists, PDFs, slides, notes and transcripts) into a study plan, and prepares one study session at a time for you to paste into an AI tutor such as ChatGPT, Claude or Gemini. It organizes the material in the background, so you can focus on studying.

## What it does

- **Ingests your material** and splits it into passages and sections, taken from each source's own structure (chapters, pages, parts).
- **Reads every passage for key ideas** and files each source into a study space, suggesting a better place when the content says so; you decide.
- **Builds a study plan** per space in the background, cached and rebuilt only when the material changes.
- **Divides shared material between topics** without an AI call. A video without chapters is one section; its passages are split between the plan's topics so that every passage belongs to exactly one topic.
- **Prepares study units** sized for one session (about 4k to 32k tokens), split into parts when a topic is too large, with the source excerpts cited by label.
- **Tells the tutor what overlaps:** a topic that repeats key ideas from an earlier source says which ideas were met before and which are new, and keeps every passage.
- **Hands off to your tutor** with a master prompt (the whole plan, your progress and how sessions work), and an optional "About you" (background, goal, practice style) added to every unit.
- **Lets your own AI arrange and steer the plan** by copy and paste, with no extra model calls: which topics to study together, in what order, where to start and what to skip.
- **Remembers where you left off**, keeps progress per topic and part, and removes material completely when you delete it, including its traces in the database file.
- **Chat and practice:** retrieval-augmented answers with citations, and quizzes with mastery tracking and spaced review.

## How it works

```
source ──► passages + embeddings ──► sections ──► key ideas
                                                     │
          study spaces ◄── placement checks (cached) ┤
                                                     ▼
          study plan (LLM, cached by content) ──► topics
                                                     │
          passage division + overlap notes (no LLM) ─┤
                                                     ▼
          study unit (or parts) ──► copied into your AI tutor
```

A few design choices that shape it:

- **Division without an LLM.** Each passage is scored against each topic with BM25 on the topic's words and with embedding similarity to a topic vector built from its best keyword matches (pseudo-relevance feedback). Embeddings are centred on the section's own mean first, because raw similarities of modern embedding models sit in a narrow band and a speaker's style dominates them. A Viterbi pass with a cost for changing topic keeps consecutive passages together.
- **Overlap from key ideas, not text similarity.** Different sources explain the same idea in their own words, so near-duplicate passages are rare across sources; shared key ideas are what reveal the overlap.
- **Few model calls.** The plan and key ideas use the Gemini API; everything else (division, overlaps, arrangement, progress) runs locally, and decisions that need judgement can go to your own chatbot through copy-and-paste requests. Temporary quota or rate-limit failures are retried later instead of being saved as failures.
- **Prompt text filled at display time.** Your tutor's name, practice style, overlaps and "About you" are added when a unit is shown, so changing them never forces prepared units to be rebuilt.

## Run it

Backend (Python 3.11 or later):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your GEMINI_API_KEY
uvicorn app.main:app --reload --port 8000
```

Web app (Node.js 18 or later), in a second terminal:

```bash
cd web
npm install
npm run dev                      # http://localhost:5173
```

Your data stays on your machine in `data/` (a SQLite database and uploaded files).

## Configuration

Models are chosen per task in `app/config.py` and can be overridden in `.env`:

```text
reasoning = gemini-3.8-flash        plans and synthesis
fast      = gemini-3.5-flash-lite   extraction, routing, part names
embedding = gemini-embedding-2
```

## Tests

```bash
python -m pytest tests -q
```

About 300 tests cover ingestion, plans, units, passage division, arrangement, progress and cleanup. They use fake AI providers, so they make no API calls.

## Project layout

| Path | What is there |
|---|---|
| `app/main.py` | HTTP API (FastAPI) |
| `app/services/` | Ingestion, plans, coverage (division and overlaps), units, handoff, arrangement, progress, cleanup, retrieval |
| `app/services/worker.py` | Background worker for ingestion and maintenance jobs |
| `web/src/` | React + TypeScript interface |
| `tests/` | pytest suite |
| `docs/` | Design notes: system design, model routing, evaluation, roadmap |
