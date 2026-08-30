from typing import Protocol

from telemachus.models import ScoredDataset


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        scored_results: list[ScoredDataset],
        task_category: str | None = None,
    ) -> list[ScoredDataset]:
        ...
