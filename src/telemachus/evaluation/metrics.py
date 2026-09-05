from math import log2

from telemachus.models import ScoredDataset


def precision_at_k(
    scored_results: list[ScoredDataset],
    relevant: set[str],
    k: int
) -> float:
    hits = sum(
        1 for res in scored_results[:k]
        if res.dataset.id in relevant
    )
    return hits / k


def recall_at_k(
    scored_results: list[ScoredDataset],
    relevant: set[str],
    k: int
) -> float:
    hits = sum(
        1 for res in scored_results[:k]
        if res.dataset.id in relevant
    )
    return hits / len(relevant) if relevant else 0.0


def reciprocal_rank(
    scored_results: list[ScoredDataset],
    relevant: set[str],
) -> float:

    for rank, res in enumerate(scored_results, start=1):
        if res.dataset.id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg(
    top_results: list[ScoredDataset],
    relevance: dict[str, int],
    k: int = 10
) -> float:

    dcg: float = 0
    for index, scored_result in enumerate(top_results[:k]):
        rel = relevance[scored_result.dataset.id]
        numerator = (2 ** (rel)) - 1
        denominator = log2(index+2)
        dcg += numerator/denominator

    idcg: float = 0
    ideal_scores = sorted(relevance.values(), reverse=True)
    for index, rel in enumerate(ideal_scores[:k]):
        numerator = (2 ** rel) - 1
        denominator = log2(index + 2)
        idcg += numerator / denominator

    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg
