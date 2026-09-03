import json
from pathlib import Path

from telemachus.evaluation.pooling import pool_candidates
from telemachus.retrieval.base import Retriever
from telemachus.retrieval.bm25 import BM25Retriever
from telemachus.retrieval.dense import DenseRetriever
from telemachus.sources.corpus import corpus_from_json


def build_candidates(retrievers: list[Retriever], queries: list[dict[str, str]], output_path: Path) -> None:

    payload = [
        {
            "query_id": query["id"],
            "candidate_ids": pool_candidates(retrievers, query["query"])
        }
        for query in queries
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False
        )


if __name__ == "__main__":
    file_path = Path("benchmarks/gold_v2/test_candidate_pools.json")
    corpus_path = Path("benchmarks/gold_v2/corpus.json")
    queries_path = Path("benchmarks/gold_v2/queries.json")
    if not file_path.is_file() or input(f"The file exists. Overwrite {file_path}? (Y/N): ").strip().lower() in ["y", "yes"]:
        if not corpus_path.is_file():
            print(f"{corpus_path} does not exist.")
        elif not queries_path.is_file():
            print(f"{queries_path} does not exist.")
        else:
            with queries_path.open("r", encoding="utf-8") as f:
                queries = json.load(f)

            corpus = corpus_from_json(corpus_path)
            retrievers: list[Retriever] = [
                BM25Retriever(corpus=corpus),
                DenseRetriever(corpus=corpus),
            ]
            build_candidates(retrievers=retrievers,
                             queries=queries,
                             output_path=file_path,
                             )
