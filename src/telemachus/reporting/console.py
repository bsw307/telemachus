from telemachus.evaluation.evaluator import EvaluationSummary
from telemachus.reporting.helpers import result_score


def print_evaluation_summary(summary: EvaluationSummary) -> None:

    border = "=" * 70
    divider = "-" * 70

    for result in summary.results:
        print()
        print(border)
        print(f'Query {result.query_id}: "{result.query}"')
        print(divider)

        print(f"NDCG@5:       {result.ndcg_5:.4f}")
        print(f"NDCG@10:      {result.ndcg_10:.4f}")
        print(f"Precision@5:  {result.precision_5:.4f}")
        print(f"Precision@10: {result.precision_10:.4f}")
        print(f"Recall@10:    {result.recall_10:.4f}")
        print(f"RR@10:        {result.reciprocal_rank:.4f}")

        print()
        print("Top Results:")
        print(divider)

        for rank, scored_result in enumerate(result.top_results[:10], start=1):
            score = result_score(scored_result)

            if score is None:
                score_text = "N/A"
            else:
                score_text = f"{score:.4f}"

            print(
                f"{rank:>2}. [{score_text:>8}] [rel={result.relevance[scored_result.dataset.id]}] "
                f"{scored_result.dataset.id}"
            )

    print()
    print(border)
    print("GLOBAL BENCHMARK EVALUATION SUMMARY")
    print(border)
    print(f"Total Queries:       {len(summary.results)}")
    print(divider)
    print(f"Mean NDCG@5:         {summary.mean_ndcg_5:.4f}")
    print(f"Mean NDCG@10:        {summary.mean_ndcg_10:.4f}")
    print(f"Mean Precision@5:    {summary.mean_precision_5:.4f}")
    print(f"Mean Precision@10:   {summary.mean_precision_10:.4f}")
    print(f"Mean Recall@10:      {summary.mean_recall_10:.4f}")
    print(f"Mean RR@10:          {summary.mean_reciprocal_rank:.4f}")
    print(border)
