from telemachus.models import ScoredDataset


def result_score(scored: ScoredDataset) -> float | None:
    if scored.final_score is not None:
        return scored.final_score

    if scored.bm25_score is not None:
        return scored.bm25_score

    return scored.dense_score
