import json
from pathlib import Path

from telemachus.evaluation.evaluator import EvaluationSummary
from telemachus.reporting.helpers import result_score


def save_evaluation(
    summary: EvaluationSummary,
    output_path: Path,
    *,
    metadata: dict[str, object],
) -> None:
    """Save a benchmark evaluation as structured JSON."""

    benchmark_output = {
        "metadata": metadata,
        "summary": {
            "num_queries": len(summary.results),
            "mean_ndcg_5": summary.mean_ndcg_5,
            "mean_ndcg_10": summary.mean_ndcg_10,
            "mean_precision_5": summary.mean_precision_5,
            "mean_precision_10": summary.mean_precision_10,
            "mean_recall_10": summary.mean_recall_10,
            "mean_reciprocal_rank": summary.mean_reciprocal_rank,
        },
        "queries": [],
    }

    for result in summary.results:
        query_output = {
            "query_id": result.query_id,
            "query": result.query,
            "metrics": {
                "ndcg_5": result.ndcg_5,
                "ndcg_10": result.ndcg_10,
                "precision_5": result.precision_5,
                "precision_10": result.precision_10,
                "recall_10": result.recall_10,
                "reciprocal_rank": result.reciprocal_rank,
            },
            "top_results": [],
        }

        for rank, scored in enumerate(result.top_results[:10], start=1):
            score = result_score(scored)

            query_output["top_results"].append(
                {
                    "rank": rank,
                    "dataset_id": scored.dataset.id,
                    "score": float(score) if score is not None else None,
                    "relevance": result.relevance[scored.dataset.id],
                }
            )

        benchmark_output["queries"].append(query_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            benchmark_output,
            f,
            indent=2,
            ensure_ascii=False,
        )
