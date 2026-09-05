from dataclasses import dataclass

from telemachus.evaluation.cases import EvaluationCase
from telemachus.evaluation.metrics import (
    ndcg,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from telemachus.models import ScoredDataset
from telemachus.reranking.base import RerankContext, Reranker
from telemachus.retrieval.base import Retriever


@dataclass
class EvaluationResult:
    query_id: str
    query: str
    ndcg_5: float
    ndcg_10: float
    precision_5: float
    precision_10: float
    recall_10: float
    reciprocal_rank: float
    top_results: list[ScoredDataset]


@dataclass
class EvaluationSummary:
    results: list[EvaluationResult]
    mean_ndcg_5: float
    mean_ndcg_10: float
    mean_precision_5: float
    mean_precision_10: float
    mean_recall_10: float
    mean_reciprocal_rank: float


def evaluate(
    retriever: Retriever,
    queries: list[dict[str, str]],
    judgments: dict[tuple[str, str], int],
    reranker: Reranker | None = None
) -> EvaluationSummary:
    eval_results: list[EvaluationResult] = []

    for query in queries:
        query_id = query["id"]
        query_text = query["query"]
        eval_results.append(evaluate_query(
            retriever,
            query_id,
            query_text,
            judgments
        ))

    mean_ndcg_5 = (
        sum(result.ndcg_5 for result in eval_results) / len(eval_results)
        if eval_results
        else 0.0
    )

    mean_ndcg_10 = (
        sum(result.ndcg_10 for result in eval_results) / len(eval_results)
        if eval_results
        else 0.0
    )
    mean_precision_5 = (
        sum(result.precision_5 for result in eval_results) / len(eval_results)
        if eval_results
        else 0.0
    )
    mean_precision_10 = (
        sum(result.precision_10 for result in eval_results) / len(eval_results)
        if eval_results
        else 0.0
    )
    mean_recall_10 = (
        sum(result.recall_10 for result in eval_results) / len(eval_results)
        if eval_results
        else 0.0
    )
    mean_reciprocal_rank = (
        sum(result.reciprocal_rank for result in eval_results) / len(eval_results)
        if eval_results
        else 0.0
    )
    return EvaluationSummary(
        results=eval_results,
        mean_ndcg_5=mean_ndcg_5,
        mean_ndcg_10=mean_ndcg_10,
        mean_precision_5=mean_precision_5,
        mean_precision_10=mean_precision_10,
        mean_recall_10=mean_recall_10,
        mean_reciprocal_rank=mean_reciprocal_rank
    )


def get_query_relevance(judgments: dict[tuple[str, str], int], query_id: str) -> dict[str, int]:
    return {
        dataset_id: relevance
        for (judgment_query_id, dataset_id), relevance in judgments.items()
        if judgment_query_id == query_id
    }


def evaluate_query(
    retriever: Retriever,
    query_id: str,
    query_text: str,
    judgments: dict[tuple[str, str], int],
) -> EvaluationResult:

    scored_results = retriever.retrieve(
        query=query_text
    )
    query_relevance = get_query_relevance(
        judgments,
        query_id
    )
    for result in scored_results[:10]:
        if result.dataset.id not in query_relevance:
            raise ValueError(f"Unjudged result: {result.dataset.id}")

    ndcg_5 = ndcg(scored_results, query_relevance, k=5)
    ndcg_10 = ndcg(scored_results, query_relevance, k=10)

    relevant_ids = {
        dataset_id
        for dataset_id, grade in query_relevance.items()
        if grade >= 1
    }

    precision_5 = precision_at_k(scored_results, relevant_ids, 5)
    precision_10 = precision_at_k(scored_results, relevant_ids, 10)
    recall_10 = recall_at_k(scored_results, relevant_ids, 10)

    rr = reciprocal_rank(scored_results[:10], relevant_ids)

    return EvaluationResult(
        query_id=query_id,
        query=query_text,
        ndcg_5=ndcg_5,
        ndcg_10=ndcg_10,
        precision_5=precision_5,
        precision_10=precision_10,
        recall_10=recall_10,
        reciprocal_rank=rr,
        top_results=scored_results
    )
