from telemachus.models import ScoredDataset


class TaskCategoryReranker:
    def __init__(self, bonus: float = 0.08):
        self.bonus = bonus

    def rerank(
        self,
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
