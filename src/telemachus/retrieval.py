from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo  # Python 3.9+ standard library

from telemachus.models import HFDatasetMetadata, ScoredDataset
from telemachus.retrieval.dense import (
    DenseRetriever,
)
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


def rerank(
    scored_datasets: list[ScoredDataset],
    task_category: str | None,
    bonus: float = 0.08,
) -> list[ScoredDataset]:
    for ds in scored_datasets:
        if (task_category is not None
            and task_category in ds.dataset.task_categories):
            ds.final_score = ds.dense_score + bonus
        else:
            ds.final_score = ds.dense_score

    return scored_datasets


@dataclass
class EvaluationCase:
    query: str
    relevant: set[str]
    task_category: str | None


def main() -> None:

    # Dummy labels for query, as determined by me.
    # Query: “dataset for training robots to perform physical manipulation tasks”
    eval_cases = [
        EvaluationCase(
            query="dataset for training robots to perform physical manipulation tasks",
            relevant={
                "genrobot2025/10Kh-RealOmin-OpenData",
                "XDOF/ABC-130k",
                "InternRobotics/InternData-A1",
                "cadene/droid",
            },
            task_category="robotics",
        ),
        EvaluationCase(
            query="English dataset for training a text classification model",
            relevant={
                "nyu-mll/glue",
                "aps/super_glue",
                "stanfordnlp/imdb",
            },
            task_category="text-classification",
        ),
        EvaluationCase(
            query="English dataset for training a question-answering model",
            relevant={
                "allenai/ai2_arc",
                "rajpurkar/squad",
            },
            task_category="question-answering",
        ),
        EvaluationCase(
            query="English legal text corpus suitable for training or fine-tuning a language model",
            relevant={
                "pile-of-law/pile-of-law",
                "mratanusarkar/Indian-Laws",
                "a2aj/canadian-case-law",
                "HFforLegal/case-law",
            },
            task_category=None,
        ),
        EvaluationCase(
            query="English medical text dataset suitable for training or fine-tuning a language model",
            relevant={
                "lavita/medical-qa-datasets",
                "medalpaca/medical_meadow_medqa",
                "medalpaca/medical_meadow_medical_flashcards",
                "medalpaca/medical_meadow_wikidoc",
                "FreedomIntelligence/medical-o1-reasoning-SFT",
            },
            task_category=None,
        ),
        EvaluationCase(
            query="English financial text corpus suitable for training or fine-tuning a language model",
            relevant={
                "artefactory/Argimi-Ardian-Finance-10k-text",
                "vidore/vidore_v3_finance_en",
                "gbharti/finance-alpaca",
            },
            task_category=None,
        ),
    ]
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

    top_k = 5
    benchmark_output = {
        "model": dense_retrieval.model_name,
        "top_k": top_k,
        "queries": [],
    }

    for case in eval_cases:

        # Score embeddings against corpus

        scored_results = dense_retrieval.retrieve(case.query)

        scored_results = rerank(
            scored_results,
            case.task_category,
        )
        scored_results.sort(
            key=lambda item: item.final_score,
            reverse=True,
        )

        top_results = scored_results[:top_k]

        # For calculating RR
        first_relevant_rank: int | None = None

        for rank, res in enumerate(scored_results, start=1):
            if res.dataset.id in case.relevant:
                first_relevant_rank = rank
                break

        rr = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0
        reciprocal_ranks.append(rr)

        # Calculate Precision@K
        hits = sum(1 for res in top_results if res.dataset.id in case.relevant)
        precision_at_k = hits / top_k
        precision_scores.append(precision_at_k)

        # Calculate Recall@K
        recall_at_k = hits / len(case.relevant)
        recall_scores.append(recall_at_k)

        # Print query benchmark breakdown
        print(f'\nQuery: "{case.query}"')
        print(f"Precision@{top_k}: {hits}/{top_k} ({precision_at_k:.0%})")
        print(
            f"Recall@{top_k}: {hits}/{len(case.relevant)} ({recall_at_k:.0%})")
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
                "precision_at_5": precision_at_k,
                "recall_at_5": recall_at_k,
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
        "num_queries": len(eval_cases),
        "mean_precision_at_5": mean_p5,
        "mean_reciprocal_rank": mrr,
        "mean_recall": mean_r5,
    }

    print("\n" + "=" * 70)
    print("GLOBAL BENCHMARK EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Test Queries:     {len(eval_cases)}")
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
