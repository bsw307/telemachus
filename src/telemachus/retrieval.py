from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo  # Python 3.9+ standard library

from telemachus.evaluation.cases import EVAL_CASES
from telemachus.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank
from telemachus.models import HFDatasetMetadata
from telemachus.reranking.task_category import TaskCategoryReranker
from telemachus.retrieval.dense import DenseRetriever
from telemachus.sources.huggingface import get_hf_datasets


# Generated
def visualize_datasets(datasets: list[HFDatasetMetadata]) -> None:
    if not datasets:
        print("No datasets found.")
        return

    print(f"\n--- Found {len(datasets)} Datasets ---")
    for idx, item in enumerate(datasets, start=1):
        desc = item.description or "No description provided."
        clean_desc = desc[:300] + "..." if len(desc) > 300 else desc

        print(f"\n{idx}. [{item.id}] ({item.name})")
        print(
            f"   Author: {item.author} | Downloads: {item.downloads:,} | byte size: {item.byte_size} size category: {item.size_category}"
        )
        print(f"   Description: {clean_desc}")
        # print(f"Semantic Representation:\n{semantic_representation(item)}")


def main() -> None:

    # Dummy labels for query, as determined by me.
    # Query: “dataset for training robots to perform physical manipulation tasks”

    # 1. Fetch datasets
    results_by_category = ["robotics",
                           "text-classification", "question-answering"]
    results_by_term = ["law", "medical", "finance"]

    raw_corpus: list[HFDatasetMetadata] = []

    for topic in results_by_category:
        raw_corpus.extend(
            get_hf_datasets(task_category=topic, lim=5, search_language="en")
        )
    for topic in results_by_term:
        raw_corpus.extend(get_hf_datasets(
            term=topic, lim=5, search_language="en"))

    seen_ids = set()
    corpus: list[HFDatasetMetadata] = []
    for ds in raw_corpus:
        if ds.id not in seen_ids:
            seen_ids.add(ds.id)
            corpus.append(ds)

    print(f"Loaded {len(corpus)} unique datasets into evaluation corpus.\n")

    reciprocal_ranks: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []

    dense_retrieval = DenseRetriever(
        corpus=corpus
    )
    task_reranker = TaskCategoryReranker(bonus=0.08)

    top_k = 5
    benchmark_output = {
        "model": dense_retrieval.model_name,
        "top_k": top_k,
        "queries": [],
    }

    for case in EVAL_CASES:

        # Score embeddings against corpus

        scored_results = dense_retrieval.retrieve(case.query)

        scored_results = task_reranker.rerank(
            scored_results,
            case.task_category,
        )
        scored_results.sort(
            key=lambda item: item.final_score,
            reverse=True,
        )

        top_results = scored_results[:top_k]
        hits = sum(1 for res in top_results if res.dataset.id in case.relevant)

        # RR for MRR
        rr = reciprocal_rank(scored_results, case.relevant)
        reciprocal_ranks.append(rr)

        # Calculate Precision@K
        precision = precision_at_k(
            top_results,
            case.relevant,
            top_k)
        precision_scores.append(precision)

        # Calculate Recall@K
        recall = recall_at_k(top_results, case.relevant)
        recall_scores.append(recall)

        # Print query benchmark breakdown
        print(f'\nQuery: "{case.query}"')
        print(f"Precision@{top_k}: {hits}/{top_k} ({precision:.0%})")
        print(
            f"Recall@{top_k}: {hits}/{len(case.relevant)} ({recall:.0%})")
        print("-" * 70)

        for rank, res in enumerate(top_results, start=1):
            is_relevant = (
                "RELEVANT" if res.dataset.id in case.relevant else "NOT RELEVANT"
            )
            print(
                f"{rank}. [{res.final_score:.4f}] {res.dataset.id:<45} Relevant: {is_relevant}"
            )
        benchmark_output["queries"].append(
            {
                "query": case.query,
                "relevant_ids": sorted(case.relevant),
                "precision_at_5": precision,
                "recall_at_5": recall,
                "reciprocal_rank": rr,
                "top_results": [
                    {
                        "rank": rank,
                        "dataset_id": result.dataset.id,
                        "score": result.final_score,
                        "relevant": result.dataset.id in case.relevant,
                    }
                    for rank, result in enumerate(top_results, start=1)
                ],
            }
        )
    mrr = sum(reciprocal_ranks) / \
        len(reciprocal_ranks) if reciprocal_ranks else 0.0
    mean_p5 = sum(precision_scores) / \
        len(precision_scores) if precision_scores else 0.0
    mean_r5 = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    benchmark_output["summary"] = {
        "num_queries": len(EVAL_CASES),
        "mean_precision_at_5": mean_p5,
        "mean_reciprocal_rank": mrr,
        "mean_recall": mean_r5,
    }

    print("\n" + "=" * 70)
    print("GLOBAL BENCHMARK EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Test Queries:     {len(EVAL_CASES)}")
    print(f"Mean Reciprocal Rank:   {mrr:.4f}")
    print(f"Mean Precision@5:       {mean_p5:.1%}")
    print(f"Mean Recall@5:       {mean_r5:.1%}")

    print("=" * 70)

    # Set your target timezone (e.g., ZoneInfo("UTC"), ZoneInfo("America/New_York"), etc.)
    tz = ZoneInfo("UTC")

    # Generate timestamp with timezone context
    timestamp = datetime.now(tz=tz).strftime("%Y%m%d_%H%M%S")
    output_path = Path(
        f"results/codespaces_reranked_semantic_{timestamp}.json")

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write benchmark output
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(benchmark_output, f, indent=2, ensure_ascii=False)


# Test / visualization
if __name__ == "__main__":
    main()
