from telemachus.models import ScoredDataset


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
