# Advanced Architecture — v0.4

LearnOS is designed around five durable assets:

1. Knowledge — source, document, section, chunk and concept structures.
2. Evidence — links from concepts and answers back to concrete source material.
3. Graph — prerequisite, related and contradiction relations.
4. State — jobs, checkpoints, study runs and learner state.
5. Evaluation — a small, repeatable benchmark that measures whether changes help.

The LLM is a replaceable reasoning component sitting on top of these assets.

## Retrieval

\`\`\`
planner
 ↓
query views
 ↓
dense + lexical candidate generation
 ↓
single RRF fusion
 ↓
graph prerequisite expansion
 ↓
single rerank
 ↓
parent / neighbour context
 ↓
answer
\`\`\`

This preserves the strengths of hybrid retrieval while allowing the agent to decide when graph traversal is justified.

## Agent

Use a small set of explicit tools and a durable state store.

The default path is deterministic. Agentic loops are reserved for cases that need additional search, comparison or verification.

## Knowledge graph

The graph begins with deterministic sequence edges and high-confidence concept relations. Prerequisite edges are directional from prerequisite to dependent concept, enabling backward traversal from a target concept.

## Hierarchy

Leaf chunks are precise retrieval units. Parent windows are context units. Retrieval and synthesis therefore use different granularities.

## Scale

SQLite remains the default local state store. Qdrant is a scale-up path for dense/sparse/multivector retrieval, introduced when measurement shows the local path has become the bottleneck.

## Evaluation

The project starts with 20–50 real questions and regression checks. New model versions, retrieval changes and graph heuristics should be tested against the same set before becoming the new default.
