from telemachus.models import ScoredDataset


def precision_at_k(top_results: list[ScoredDataset], relevant: set[str], top_k: int) -> float:
    hits = sum(1 for res in top_results if res.dataset.id in relevant)
    return hits / top_k


def recall_at_k(top_results: list[ScoredDataset], relevant: set[str]) -> float:
    hits = sum(1 for res in top_results if res.dataset.id in relevant)
    return hits / len(relevant)


def reciprocal_rank(scored_results: list[ScoredDataset], relevant: set[str]) -> float:

    for rank, res in enumerate(scored_results, start=1):
        if res.dataset.id in relevant:
            return 1.0 / rank

    return 0.0
