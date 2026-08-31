from dataclasses import dataclass

from telemachus.evaluation.cases import EvaluationCase
from telemachus.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank
from telemachus.models import ScoredDataset
from telemachus.reranking.base import RerankContext, Reranker
from telemachus.retrieval.base import Retriever


@dataclass
class EvaluationResult:
    query: str
    relevant: set[str]
    precision: float
    recall: float
    reciprocal_rank: float
    top_results: list[ScoredDataset]


@dataclass
class EvaluationSummary:
    results: list[EvaluationResult]
    mean_precision: float
    mean_recall: float
    mean_reciprocal_rank: float


def evaluate(
    retriever: Retriever,
    evaluation_cases: list[EvaluationCase],
    reranker: Reranker | None = None,
    *,
    top_k: int = 5
) -> EvaluationSummary:

    reciprocal_ranks: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []

    eval_results: list[EvaluationResult] = []

    for case in evaluation_cases:

        # Score embeddings against corpus
        scored_results = retriever.retrieve(case.query)

        if reranker is not None:
            context = RerankContext(
                task_category=case.task_category,
            )
            scored_results = reranker.rerank(
                case.query,
                scored_results,
                context=context
            )

        top_results = scored_results[:top_k]

        # RR for MRR
        rr = reciprocal_rank(scored_results, case.relevant)
        reciprocal_ranks.append(rr)

        # Calculate Precision@K
        precision = precision_at_k(
            top_results,
            case.relevant,
            top_k)
        precision_scores.append(precision)

        # Calculate Recall@K
        recall = recall_at_k(top_results, case.relevant)
        recall_scores.append(recall)

        eval_results.append(
            EvaluationResult(
                query=case.query,
                relevant=case.relevant,
                precision=precision,
                recall=recall,
                reciprocal_rank=rr,
                top_results=top_results
            ))
    mrr = sum(reciprocal_ranks) / \
        len(reciprocal_ranks) if reciprocal_ranks else 0.0
    mean_precision = sum(precision_scores) / \
        len(precision_scores) if precision_scores else 0.0
    mean_recall = sum(recall_scores) / \
        len(recall_scores) if recall_scores else 0.0

    return EvaluationSummary(
        results=eval_results,
        mean_precision=mean_precision,
        mean_recall=mean_recall,
        mean_reciprocal_rank=mrr
    )
