# Telemachus

A small retrieval experiment for searching Hugging Face datasets with **semantic embeddings + structured metadata**.

## What it does

```text
Hugging Face dataset cards
        ↓
title + description embeddings
        ↓
cosine similarity
        ↓
task-aware reranking
        ↓
Precision@5 / Recall@5 / MRR
```

The baseline uses only semantic similarity. A second version adds a small bonus when a dataset's Hugging Face task category matches the query.

## Results

| Method              | Precision@5 |  Recall@5 |       MRR |
| ------------------- | ----------: | --------: | --------: |
| Semantic baseline   |       63.3% |     88.9% |     0.917 |
| Task-aware reranker |   **66.7%** | **94.4%** | **1.000** |

The clearest improvement was on text-classification queries, where semantic search retrieved related NLP datasets that were not actually classification datasets.

This is a very, very small experiment, mostly meant for my own practice. If I dare infer anything it's that embeddings are good at finding semantically related candidates (duh); structured metadata can help distinguish which candidates are actually suitable.

## Evaluation

The benchmark uses:

* 29 Hugging Face datasets
* 6 manually labeled queries
* Precision@5
* Recall@5
* Mean Reciprocal Rank

The embedding model is `sentence-transformers/all-MiniLM-L12-v2`.

## Run

```bash
uv sync
uv run python -m telemachus.retrieval
```

Saved experiment outputs are in `results/`.

## Limitations

I ran this on my x86_64 intel mac, meaning I had to pin torch and numpy to older versions.

This is a small prototype, not a production benchmark. The corpus and query set are limited, relevance labels are manual, and the reranking weight is hand-selected. etc. 

Possible extensions include learned reranking, additional metadata signals, hybrid retrieval, and knowledge-graph relationships.
