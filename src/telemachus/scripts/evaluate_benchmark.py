import json
from pathlib import Path

from telemachus.evaluation.evaluator import EvaluationSummary, evaluate
from telemachus.evaluation.judgments import load_judgments
from telemachus.reporting.console import print_evaluation_summary
from telemachus.reporting.json_output import save_evaluation
from telemachus.retrieval.bm25 import BM25Retriever
from telemachus.retrieval.dense import DenseRetriever
from telemachus.sources.corpus import corpus_from_json


def evaluate_gold_v2(
    retriever_type,
    benchmark_dir: Path,
) -> EvaluationSummary:

    # Construct paths
    corpus_path = benchmark_dir / "corpus.json"
    judgments_path = benchmark_dir / "judgments.jsonl"
    queries_path = benchmark_dir / "queries.json"

    # Load
    corpus = corpus_from_json(corpus_path)
    judgments = load_judgments(judgments_path)
    with queries_path.open("r", encoding="utf-8") as f:
        queries = json.load(f)

    retriever = retriever_type(corpus)

    return evaluate(
        retriever=retriever,
        queries=queries,
        judgments=judgments
    )


if __name__ == "__main__":
    benchmark_path = Path("benchmarks/gold_v2")
    retriever = BM25Retriever
    summary = evaluate_gold_v2(retriever, benchmark_path)
    print_evaluation_summary(summary)
    save_evaluation(
        summary,
        Path("results/gold_v2/bm25.json"),
        metadata={
            "benchmark": "gold_v2",
            "retriever": "bm25",
            "model": "sentence-transformers/all-MiniLM-L12-v2",
        },
    )
