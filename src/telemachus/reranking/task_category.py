from telemachus.models import ScoredDataset
from telemachus.reranking.base import RerankContext


class TaskCategoryReranker:
    def __init__(self, bonus: float = 0.08):
        self.bonus = bonus

    def rerank(
        self,
        query: str,
        scored_results: list[ScoredDataset],
        context: RerankContext | None = None,
    ) -> list[ScoredDataset]:
        task_category = (
            context.task_category
            if context is not None
            else None
        )

        for result in scored_results:
            if (task_category is not None
                    and task_category in result.dataset.task_categories):
                result.final_score = result.dense_score + self.bonus
            else:
                result.final_score = result.dense_score

        scored_results.sort(
            key=lambda item: item.final_score,
            reverse=True,
        )

        return scored_results
