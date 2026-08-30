from telemachus.evaluation.metrics import (
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

    return ScoredDataset(
        dataset=dataset,
        dense_score=0.0,
    )


def test_precision_at_k():
    results = [
        make_result("X"),
        make_result("B"),
        make_result("Y"),
        make_result("A"),
        make_result("Z"),
    ]

    relevant = {"A", "B", "C"}

    assert precision_at_k(results, relevant, 5) == 0.4


def test_precision_at_k_with_no_hits():
    results = [
        make_result("X"),
        make_result("Y"),
        make_result("Z"),
    ]

    relevant = {"A", "B"}

    assert precision_at_k(results, relevant, 3) == 0.0


def test_recall_at_k():
    results = [
        make_result("X"),
        make_result("B"),
        make_result("Y"),
        make_result("A"),
        make_result("Z"),
    ]

    relevant = {"A", "B", "C"}

    assert recall_at_k(results, relevant) == 2 / 3


def test_recall_at_k_with_no_hits():
    results = [
        make_result("X"),
        make_result("Y"),
    ]

    relevant = {"A", "B"}

    assert recall_at_k(results, relevant) == 0.0


def test_reciprocal_rank_first_result():
    results = [
        make_result("A"),
        make_result("X"),
        make_result("Y"),
    ]

    relevant = {"A"}

    assert reciprocal_rank(results, relevant) == 1.0


def test_reciprocal_rank_second_result():
    results = [
        make_result("X"),
        make_result("A"),
        make_result("Y"),
    ]

    relevant = {"A"}

    assert reciprocal_rank(results, relevant) == 0.5


def test_reciprocal_rank_with_no_relevant_result():
    results = [
        make_result("X"),
        make_result("Y"),
        make_result("Z"),
    ]

    relevant = {"A", "B"}

    assert reciprocal_rank(results, relevant) == 0.0


def test_metrics_on_known_ranking():
    results = [
        make_result("X"),
        make_result("B"),
        make_result("Y"),
        make_result("A"),
        make_result("Z"),
    ]

    relevant = {"A", "B", "C"}

    assert precision_at_k(results, relevant, 5) == 0.4
    assert recall_at_k(results, relevant) == 2 / 3
    assert reciprocal_rank(results, relevant) == 0.5