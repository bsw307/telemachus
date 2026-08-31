from dataclasses import dataclass
from typing import Protocol

from telemachus.models import ScoredDataset


@dataclass
class RerankContext:
    task_category: str | None = None


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        scored_results: list[ScoredDataset],
        context: RerankContext | None = None,
    ) -> list[ScoredDataset]:
        ...
