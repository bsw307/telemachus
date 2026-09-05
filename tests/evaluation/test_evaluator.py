from pytest import approx
from test_metrics import make_result

from telemachus.evaluation.evaluator import evaluate, get_query_relevance
from telemachus.models import ScoredDataset


class FakeRetriever:
    def retrieve(self, query: str, k: int | None = None) -> list[ScoredDataset]:
        results = [
            make_result("A"),
            make_result("B"),
            make_result("C"),
        ]
        return results if k is None else results[:k]


def test_evaluator():
    queries = [
        {"id": "q001", "query": "A"},

    ]
    judgments = {
        ("q001", "A"): 2,
        ("q001", "B"): 1,
        ("q001", "C"): 0,
    }
    test_retriever = FakeRetriever()
    test_summary = evaluate(test_retriever, queries, judgments)

    assert len(test_summary.results) == 1

    result = test_summary.results[0]

    assert result.query_id == "q001"
    assert result.query == "A"

    assert result.ndcg_5 == approx(1.0)
    assert result.ndcg_10 == approx(1.0)

    assert result.precision_5 == approx(0.4)
    assert result.precision_10 == approx(0.2)
    assert result.recall_10 == approx(1.0)

    assert result.reciprocal_rank == approx(1.0)

    assert test_summary.mean_ndcg_5 == approx(1.0)
    assert test_summary.mean_ndcg_10 == approx(1.0)

    assert test_summary.mean_precision_5 == approx(0.4)
    assert test_summary.mean_precision_10 == approx(0.2)
    assert test_summary.mean_recall_10 == approx(1.0)

    assert test_summary.mean_reciprocal_rank == approx(1.0)


def test_get_query_relevance():
    judgments = {
        ("q001", "A"): 2,
        ("q001", "B"): 1,
        ("q002", "A"): 0,
    }

    result = get_query_relevance(judgments, "q001")

    assert result == {
        "A": 2,
        "B": 1,
    }
