# Evaluation Plan

## Retrieval set

Create 20–50 questions for a tiny corpus first. Label expected supporting chunk IDs manually.

## Retrieval metrics

- Recall@5
- Recall@10
- MRR
- first relevant rank
- percentage of answers with a correct citation

## Graph metrics

- prerequisite edge precision
- useful prerequisite traversal rate
- irrelevant traversal rate
- cycle detection
- orphan concept rate

## Organization metrics

- group assignment accuracy
- duplicate concept rate
- concept coverage
- percentage of orphan chunks

## Workflow metrics

- successful resume after forced shutdown
- no duplicate embeddings after retry
- no duplicate chunks after re-upload
- retry recovery from HTTP 429/5xx
- no duplicate relation creation on rerun

## Agent metrics

Record study traces with:

```text
run_id
task
model
query_plan
selected_groups
target_concepts
retrieved_chunk_ids
latency
result_counts
```

For difficult agentic workflows, evaluate the full trace, not only the final answer.

## Cost/latency metrics

Track:

```text
source
chunks
embedding_batches
LLM_calls
LLM_task_breakdown
processing_seconds
```

## Regression policy

Run the same benchmark whenever any of these change:

- model version
- prompt/schema
- chunking
- retrieval candidate depth
- reranking
- graph extraction
- graph traversal
- curriculum logic

This makes model upgrades measurable instead of subjective.
