# Model Routing

Model choice is a task policy, not an application-wide constant.

| Task | Primary | Fallback | Why |
|---|---|---|---|
| Source profile | Gemini 3.5 Flash-Lite | 3.8 Flash | high-volume classification |
| Group routing | Gemini 3.5 Flash-Lite | 3.8 Flash | cheap ambiguity resolution |
| Concept extraction | Gemini 3.5 Flash-Lite | 3.8 Flash | structured extraction |
| Relationship extraction | Gemini 3.5 Flash-Lite | 3.8 Flash | graph construction |
| Query planning | Gemini 3.5 Flash-Lite | 3.8 Flash | compact planning |
| Reranking | Gemini 3.5 Flash-Lite | 3.8 Flash | candidate scoring |
| Syllabus | Gemini 3.8 Flash | 3.5 Flash-Lite | global curriculum reasoning |
| Study synthesis | Gemini 3.8 Flash | 3.5 Flash-Lite | teaching and synthesis |
| Difficult agent planning | Gemini 3.8 Flash | 3.5 Flash-Lite | multi-step reasoning |
| Embeddings | Gemini Embedding 2 | provider fallback | semantic indexing |

The provider layer must remain replaceable. A future OpenAI or Anthropic adapter should fit the same task interface rather than leaking provider-specific assumptions into retrieval or domain code.

Structured outputs are preferred for planner and graph decisions so the application, not prose parsing, controls tool selection and state changes.

Model changes should be validated against the project's retrieval and learning eval suite before adoption.
