# Telemachus

Telemachus is an experimental retrieval framework for studying agentic retrieval workflows over semi-structured data.

The project explores whether an agent can improve retrieval by deciding how to search, rather than relying on a single fixed retrieval strategy. Planned agentic behaviors include query rewriting, query decomposition, iterative retrieval, result critique, and dynamic selection between lexical, dense, and hybrid search.

The current implementation establishes the retrieval and evaluation infrastructure needed to test those strategies rigorously. Hugging Face dataset cards are used as the initial corpus because they provide heterogeneous metadata, overlapping domains, ambiguous descriptions, and realistic retrieval failures.

## Goal

A conventional retrieval pipeline looks roughly like:

```text id="10tf4v"
query
  ↓
fixed retriever
  ↓
ranked results
```

Telemachus is moving toward:

```text id="vpp2md"
                ┌───────────────┐
query ─────────>│ retrieval     │
                │ agent         │
                └───────┬───────┘
                        │
         ┌──────────────┼──────────────┐
         ↓              ↓              ↓
       BM25           Dense          Hybrid
         │              │              │
         └──────────────┼──────────────┘
                        ↓
                  inspect results
                        ↓
              rewrite / decompose /
              retrieve again / stop
                        ↓
                   final ranking
```

The central research question is:

> Can an agent make better retrieval decisions than a fixed retrieval pipeline, and is the improvement worth the additional latency and cost?

The goal is therefore not simply to add an LLM to search, but to make agentic retrieval **measurable against simpler baselines**.

## Current Status

The current checkpoint provides the deterministic retrieval foundation for future agentic experiments:

* Hugging Face dataset metadata ingestion
* Shared textual representation of dataset cards
* Dense semantic retrieval using Sentence Transformers
* BM25 lexical retrieval using `bm25s`
* Common retriever interface
* Optional reranking interface
* Task-category reranking
* Precision@K, Recall@K, and Mean Reciprocal Rank evaluation
* Console and JSON benchmark reporting
* Unit tests for models, metrics, and BM25 retrieval

This makes retrieval strategies interchangeable while keeping the corpus, queries, relevance judgments, and metrics constant.

## Agentic Retrieval Roadmap

The next stages will progressively give an agent more control over retrieval.

### 1. Query rewriting

The first agentic experiment will allow a model to rewrite a user's query before retrieval.

```text id="14soez"
original query
      ↓
     LLM
      ↓
rewritten query
      ↓
retriever
```

The rewritten-query strategy can then be directly compared with retrieving the original query.

### 2. Query decomposition

For queries containing multiple constraints or information needs, the agent can generate several focused retrieval queries.

```text id="y3mybs"
"English medical QA data for LLM fine-tuning"
                  ↓
               agent
          ┌───────┼────────┐
          ↓       ↓        ↓
      medical    QA     English /
                         training
          └───────┼────────┘
                  ↓
           merge candidates
```

### 3. Iterative retrieval

Rather than performing one search, an agent can inspect the first retrieval result and decide whether another retrieval step is necessary.

```text id="c1mr0f"
query
  ↓
retrieve
  ↓
inspect candidates
  ↓
sufficient? ── yes ──> return
    │
    no
    ↓
rewrite / refine
    ↓
retrieve again
```

### 4. Dynamic retriever selection

Because Telemachus exposes BM25 and dense retrieval through the same interface, an agent can eventually choose the search strategy based on the query.

For example:

```text id="xr6brg"
exact terminology / named entity
        ↓
       BM25

semantic intent
        ↓
       Dense

mixed constraints
        ↓
      Hybrid
```

This can later be extended to additional retrieval methods such as graph traversal.

### 5. Retrieval traces and agent evaluation

Agentic systems introduce dimensions that retrieval metrics alone do not capture.

Future experiments will therefore record:

```text id="s0k3xu"
retrieval quality
latency
model/API cost
number of retrieval steps
retriever chosen
query rewrites
intermediate decisions
final ranking
```

This will make it possible to evaluate not only whether an agent improves retrieval, but **how it achieved that improvement and what it cost**.

## Architecture

```text id="dh248c"
Dataset source
     ↓
HFDatasetMetadata
     ↓
dataset_text()
     ↓
┌───────────────────────┐
│ Retrieval strategies  │
│                       │
│ BM25Retriever         │
│ DenseRetriever        │
│ future HybridRetriever│
└───────────────────────┘
     ↓
ScoredDataset
     ↓
optional reranker
     ↓
shared evaluator
     ↓
metrics + reporting
```

The future agent layer will orchestrate these components rather than replacing them:

```text id="p5deor"
                   Retrieval Agent
                         ↓
          strategy / query decisions
                         ↓
              Retriever interface
              /        |        \
           BM25      Dense      Hybrid
              \        |        /
                    candidates
                         ↓
                     reranker
                         ↓
                    evaluator
```

This separation is intentional: agentic strategies can be evaluated against the exact same deterministic retrieval components used for the baselines.

## Baseline Experiment

The current development benchmark contains 29 Hugging Face dataset cards and six retrieval queries.

### Dense retrieval

Using `sentence-transformers/all-MiniLM-L12-v2`, without reranking:

| Metric               |  Score |
| -------------------- | -----: |
| Mean Reciprocal Rank | 0.9167 |
| Precision@5          |  56.7% |
| Recall@5             |  77.2% |

### BM25

Using BM25 without reranking:

| Metric               |  Score |
| -------------------- | -----: |
| Mean Reciprocal Rank | 0.5417 |
| Precision@5          |  23.3% |
| Recall@5             |  36.1% |

Dense retrieval performed substantially better on the current benchmark, especially for semantic queries such as robot manipulation and medical-data discovery.

The comparison also exposed weaknesses in the current relevance judgments. Several retrieved datasets appear plausibly relevant despite not being included in the manually defined gold sets. The current numbers should therefore be treated as **development baselines**, not definitive retrieval-quality measurements.

That limitation motivates the next evaluation milestone: a larger fixed corpus and pooled manual relevance judgments.

## Evaluation Strategy

Agentic retrieval is only useful if its improvements can be measured against simpler alternatives.

The planned benchmark will use:

```text id="4drms6"
fixed corpus
    +
fixed queries
    +
human relevance judgments
    ↓
same evaluator
    ↓
BM25
Dense
Hybrid
Agentic
```

Candidate documents will be pooled from multiple retrieval systems and manually assigned graded relevance judgments.

This is intended to prevent improvements from being attributed to an agent when they are actually caused by incomplete or inconsistent evaluation data.

## Project Structure

```text id="y82f2f"
src/telemachus/
├── evaluation/
│   ├── cases.py
│   ├── evaluator.py
│   └── metrics.py
├── representations/
│   └── dataset.py
├── reporting/
│   ├── console.py
│   ├── helpers.py
│   └── json_output.py
├── reranking/
│   ├── base.py
│   └── task_category.py
├── retrieval/
│   ├── base.py
│   ├── bm25.py
│   └── dense.py
├── sources/
│   └── huggingface.py
└── models.py
```

## Running

Install dependencies:

```bash id="qodnmk"
uv sync
```

Run tests:

```bash id="vqk71a"
uv run pytest
```

Run linting:

```bash id="nmuw6k"
uv run ruff check .
```

Run the current benchmark:

```bash id="06de06"
uv run python retrieval.py
```

## Near-Term Priorities

1. Build a larger fixed evaluation corpus.
2. Create pooled manual relevance judgments.
3. Implement hybrid BM25 + dense retrieval.
4. Add the first agentic experiment: **query rewriting**.
5. Add query decomposition.
6. Add iterative retrieve → inspect → retrieve workflows.
7. Let the agent dynamically choose between retrieval strategies.
8. Record quality, latency, cost, and retrieval traces for each strategy.

The guiding principle is to introduce agentic behavior incrementally and require each additional layer of complexity to demonstrate measurable value over simpler retrieval baselines.
