# Gemini Model Routing

| Task | Primary | Fallback | Reason |
|---|---|---|---|
| Source profile | Gemini 3.5 Flash-Lite | 3.5 Flash-Lite | High-volume classification |
| Group routing | Gemini 3.5 Flash-Lite | 3.8 Flash | Ambiguous semantic routing |
| Concept extraction | Gemini 3.5 Flash-Lite | 3.8 Flash | Structured extraction |
| Query planning | Gemini 3.5 Flash-Lite | 3.8 Flash | Small routing problem |
| Reranking | Gemini 3.5 Flash-Lite | 3.8 Flash | Candidate scoring |
| Syllabus | Gemini 3.8 Flash | 3.5 Flash-Lite | Global curriculum reasoning |
| Study pack | Gemini 3.8 Flash | 3.5 Flash-Lite | Synthesis and teaching |
| Final answer | Gemini 3.8 Flash | 3.5 Flash-Lite | Higher reasoning demand |

The API client uses `generateContent` with structured JSON output schemas for machine-readable decisions and explicit `thinkingLevel` settings. This is supported by the current Gemini API.
