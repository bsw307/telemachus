from typing import Protocol

from telemachus.models import ScoredDataset


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        k: int | None = None
    ) -> list[ScoredDataset]:
        ...
