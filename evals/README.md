# LearnOS evaluation set

Start with 20–50 questions taken from real study failures.

Each JSONL row:

```json
{"question":"Why does PCA require eigenvectors?","course_id":1,"relevant_ids":[42,57]}
```

The evaluation runner should record retrieved chunk IDs and then compute Recall@5, Recall@10 and MRR.

Keep this set small and real at first. When model, retrieval, chunking or graph logic changes, run it as a regression suite before importing a larger corpus.

Recommended additional checks:

- prerequisite traversal reaches a useful prerequisite without unrelated expansion;
- citations point to evidence actually present in the selected chunks;
- adding a new source does not destroy existing concept identity;
- retrying a failed ingestion does not duplicate chunks or embeddings.
