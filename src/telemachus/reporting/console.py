from telemachus.evaluation.evaluator import EvaluationSummary


def print_evaluation_summary(
    summary: EvaluationSummary,
    top_k: int,
) -> None:

    for case in summary.results:
        # Print query benchmark breakdown

        print(f'\nQuery: "{case.query}"')
        print(f"Precision@{case.precision}: ({case.precision:.0%})")
        print(
            f"Recall@{top_k}: ({case.recall:.0%})")
        print("-" * 70)

        for rank, res in enumerate(case.top_results, start=1):
            is_relevant = (
                "RELEVANT" if res.dataset.id in case.relevant else "NOT RELEVANT"
            )
            print(
                f"{rank}. [{res.final_score:.4f}] {res.dataset.id:<45} Relevant: {is_relevant}"
            )

    print("\n" + "=" * 70)
    print("GLOBAL BENCHMARK EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Test Queries:     {len(summary.results)}")
    print(f"Mean Reciprocal Rank:   {summary.mean_reciprocal_rank:.4f}")
    print(f"Mean Precision@5:       {summary.mean_precision:.1%}")
    print(f"Mean Recall@5:       {summary.mean_recall:.1%}")

    print("=" * 70)
