import json
from pathlib import Path

from telemachus.evaluation.evaluator import EvaluationSummary
from telemachus.reporting.helpers import result_score


def save_evaluation(
    summary: EvaluationSummary,
    output_path: Path,
    *,
    metadata: dict[str, object],
    top_k: int,
) -> None:

    benchmark_output = {
        "metadata": metadata,
        "top_k": top_k,
        "queries": [],
        "summary": {
            "num_queries": len(summary.results),
            "mean_precision": summary.mean_precision,
            "mean_recall": summary.mean_recall,
            "mean_reciprocal_rank": summary.mean_reciprocal_rank,
        },
    }
    for result in summary.results:
        benchmark_output["queries"].append(
            {
                "query": result.query,
                "relevant_ids": sorted(result.relevant),
                f"precision_at_{top_k}": result.precision,
                f"recall_at_{top_k}": result.recall,
                "reciprocal_rank": result.reciprocal_rank,
                "top_results": [
                    {
                        "rank": rank,
                        "dataset_id": scored.dataset.id,
                        "score": (
                            result_score(scored)
                        ),
                        "relevant": scored.dataset.id in result.relevant,
                    }
                    for rank, scored in enumerate(
                        result.top_results,
                        start=1,
                    )
                ],
            }
        )

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write benchmark output
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            benchmark_output,
            f,
            indent=2,
            ensure_ascii=False
        )
