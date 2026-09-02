from telemachus.retrieval.base import Retriever


def pool_candidates(
        retrievers: list[Retriever],
        query: str,
        k: int = 10
) -> list[str]:
    candidate_ids: set[str] = set()
    for retriever in retrievers:
        results = retriever.retrieve(query=query, k=k)
        for result in results:
            candidate_ids.add(result.dataset.id)
    return sorted(candidate_ids)
