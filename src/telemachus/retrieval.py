from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from telemachus.evaluation.cases import EVAL_CASES
from telemachus.evaluation.evaluator import evaluate
from telemachus.models import HFDatasetMetadata
from telemachus.reporting.console import print_evaluation_summary
from telemachus.reporting.json_output import save_evaluation
from telemachus.reranking.task_category import TaskCategoryReranker
from telemachus.retrieval.dense import DenseRetriever
from telemachus.sources.huggingface import get_hf_datasets


def main() -> None:

    results_by_category = ["robotics",
                           "text-classification",
                           "question-answering"]

    results_by_term = ["law", "medical", "finance"]

    raw_corpus: list[HFDatasetMetadata] = []

    for topic in results_by_category:
        raw_corpus.extend(
            get_hf_datasets(
                task_category=topic,
                lim=5,
                search_language="en"
            )
        )
    for topic in results_by_term:
        raw_corpus.extend(
            get_hf_datasets(
                term=topic,
                lim=5,
                search_language="en"
            )
        )

    seen_ids = set()
    corpus: list[HFDatasetMetadata] = []
    for ds in raw_corpus:
        if ds.id not in seen_ids:
            seen_ids.add(ds.id)
            corpus.append(ds)

    print(f"Loaded {len(corpus)} unique datasets into evaluation corpus.\n")

    retriever = DenseRetriever(corpus=corpus)
    reranker = TaskCategoryReranker(bonus=0.08)

    k = 5

    metadata: dict[str, object] = {
        "retriever": "dense",
        "retriever_config": {
            "model": retriever.model_name
        },
        "reranker": "task_category",
        "reranker_config": {
            "bonus": reranker.bonus
        },
    }

    summary = evaluate(
        retriever,
        EVAL_CASES,
        reranker=reranker,
        top_k=k
    )
    print_evaluation_summary(summary=summary, top_k=k)

    tz = ZoneInfo("UTC")
    timestamp = datetime.now(tz=tz).strftime("%Y%m%d_%H%M%S")

    output_path = Path(
        f"results/codespaces_reranked_semantic_{timestamp}.json"
    )

    save_evaluation(
        summary=summary,
        output_path=output_path,
        metadata=metadata,
        top_k=k
    )


# Test / visualization
if __name__ == "__main__":
    main()
