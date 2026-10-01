# Evaluation Plan

## Retrieval set

Create 20-50 questions for a tiny corpus first. Label the expected supporting chunk IDs manually.

## Retrieval metrics

- Recall@5
- Recall@10
- MRR
- rank position of the first relevant chunk
- percentage of answers with at least one correct citation

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

The expected optimization is to make embedding batch calls and group-level concept extraction dominate the background pipeline rather than one LLM call per chunk.
