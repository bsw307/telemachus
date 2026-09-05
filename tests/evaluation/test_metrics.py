import pytest
from pytest import approx

from telemachus.evaluation.metrics import (
    ndcg,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from telemachus.models import HFDatasetMetadata, ScoredDataset


def make_result(dataset_id: str) -> ScoredDataset:
    dataset = HFDatasetMetadata(
        id=dataset_id,
        name=dataset_id,
        author="test",
    )
    return ScoredDataset(dataset=dataset)


def test_precision_at_k():
    results = [
        make_result("X"),
        make_result("B"),
        make_result("Y"),
        make_result("A"),
        make_result("Z"),
    ]

    relevant = {"A", "B", "C"}

    assert precision_at_k(results, relevant, 5) == approx(0.4)


def test_precision_at_k_with_no_hits():
    results = [
        make_result("X"),
        make_result("Y"),
        make_result("Z"),
    ]

    relevant = {"A", "B"}

    assert precision_at_k(results, relevant, 3) == approx(0.0)


def test_recall_at_k():
    results = [
        make_result("X"),
        make_result("B"),
        make_result("Y"),
        make_result("A"),
        make_result("Z"),
    ]

    relevant = {"A", "B", "C"}

    assert recall_at_k(results, relevant) == approx(2 / 3)


def test_recall_at_k_with_no_hits():
    results = [
        make_result("X"),
        make_result("Y"),
    ]

    relevant = {"A", "B"}

    assert recall_at_k(results, relevant) == approx(0.0)


def test_reciprocal_rank_first_result():
    results = [
        make_result("A"),
        make_result("X"),
        make_result("Y"),
    ]

    relevant = {"A"}

    assert reciprocal_rank(results, relevant) == approx(1.0)


def test_reciprocal_rank_second_result():
    results = [
        make_result("X"),
        make_result("A"),
        make_result("Y"),
    ]

    relevant = {"A"}

    assert reciprocal_rank(results, relevant) == approx(0.5)


def test_reciprocal_rank_with_no_relevant_result():
    results = [
        make_result("X"),
        make_result("Y"),
        make_result("Z"),
    ]

    relevant = {"A", "B"}

    assert reciprocal_rank(results, relevant) == approx(0.0)


def test_metrics_on_known_ranking():
    results = [
        make_result("X"),
        make_result("B"),
        make_result("Y"),
        make_result("A"),
        make_result("Z"),
    ]

    relevant = {"A", "B", "C"}

    assert precision_at_k(results, relevant, 5) == approx(0.4)
    assert recall_at_k(results, relevant) == approx(2 / 3)
    assert reciprocal_rank(results, relevant) == approx(0.5)


def test_ndcg_perfect_ranking():
    relevance = {
        "A": 2,
        "B": 2,
        "C": 1,
        "D": 0
    }
    results = [make_result(key) for key in relevance]

    score = ndcg(results, relevance, k=4)

    assert score == approx(1.0)


def test_bad_ndcg_ranking():
    relevance = {
        "A": 1,
        "B": 2,
        "C": 1,
        "D": 3
    }
    results = [make_result(key) for key in relevance]

    score = ndcg(results, relevance, k=4)

    assert score < 1.0


def test_ndcg_unjudged():
    relevance = {
        "A": 2,
        "B": 1,
    }

    results = [
        make_result("A"),
        make_result("UNJUDGED"),
    ]

    with pytest.raises(KeyError):
        ndcg(results, relevance)


def test_ndcg_top_k():
    relevance = {
        "A": 2,
        "B": 1,
        "C": 0,
        "D": 2,
    }
    results = [make_result(key) for key in relevance]

    score = ndcg(results, relevance, k=1)

    assert score == approx(1.0)


def test_ndcg_no_relevant_results():
    relevance = {
        "A": 0,
    }
    results = [make_result(key) for key in relevance]

    score = ndcg(results, relevance)

    assert score == 0
