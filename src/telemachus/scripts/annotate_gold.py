import json
import re
import textwrap
from pathlib import Path

from telemachus.models import HFDatasetMetadata
from telemachus.sources.corpus import corpus_from_json


def format_description(description: str | None, width: int = 88) -> str:
    if not description:
        return "No description available."

    text = description

    text = text.replace(
        "Your browser does not support the video tag.",
        "",
    )

    text = re.sub(
        r"See the full description on the dataset page: https?://\S+\.?",
        "",
        text,
    )

    text = " ".join(text.split())

    return textwrap.fill(text, width=width)


def display_candidate(query_text: str,
                      dataset: HFDatasetMetadata,
                      query_index: int,
                      query_total: int,
                      candidate_index: int,
                      candidate_total: int,
                      title: str = "Gold v2 Annotation"
                      ) -> None:
    width = 88
    progress = (
        f"Query {query_index}/{query_total} | " f"Candidate {candidate_index}/{candidate_total}")
    print("\033[2J\033[H", end="")
    description = format_description(dataset.description, width=width)
    spacing = max(1, width - len(title) - len(progress))

    # Title
    print("="*width)
    print(title + " " * spacing + progress)
    print("="*width)

    # Query
    print(f"\nQuery\n{query_text}\n")

    # Dataset ID
    print("-"*width)
    print(f"ID:   {dataset.id}")
    print(f"Name: {dataset.name}")
    print("-"*width)

    # Tasks and language
    tasks = ", ".join(dataset.task_categories) or "—"
    languages = ", ".join(dataset.language) or "—"
    print(f"Tasks:     {tasks}")
    print(f"Languages: {languages}\n")

    # Description
    print(description)
    print("\n")
    print("-"*width)

    # Instructions
    print("\n2 = highly/directly relevant")
    print("1 = partially relevant")
    print("0 = irrelevant")
    print("s = skip | q = quit\n")


def annotate(queries_path: Path, candidate_pool: Path, corpus_path: Path, output_path: Path) -> None:

    with queries_path.open("r", encoding="utf-8") as f:
        queries = json.load(f)
    with candidate_pool.open("r", encoding="utf-8") as f:
        candidates = json.load(f)

    existing_judgments = []

    if output_path.is_file():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line:
                    existing_judgments.append(json.loads(line))

    queries_by_id = {
        query["id"]: query
        for query in queries
    }
    corpus = corpus_from_json(corpus_path)

    judged_pairs = {
        (judgment["query_id"], judgment["dataset_id"])
        for judgment in existing_judgments
    }

    datasets_by_id = {
        dataset.id: dataset
        for dataset in corpus
    }

    for query_index, entry in enumerate(candidates, start=1):
        query_id = entry["query_id"]
        query_text = queries_by_id[query_id]["query"]

        for candidate_index, candidate_id in enumerate(entry["candidate_ids"], start=1):
            if (query_id, candidate_id) in judged_pairs:
                continue
            dataset = datasets_by_id[candidate_id]

            display_candidate(
                query_text=query_text,
                dataset=dataset,
                query_index=query_index,
                query_total=len(candidates),
                candidate_index=candidate_index,
                candidate_total=len(entry["candidate_ids"]),
            )
            while True:
                judgment = input("Relevance: ").strip().lower()

                if judgment in {"0", "1", "2", "s", "q", "quit"}:
                    break

                print("Please enter 2, 1, 0, s, or q.")

            if judgment in {"0", "1", "2"}:
                judgment_record = {
                    "query_id": query_id,
                    "dataset_id": dataset.id,
                    "relevance": int(judgment)
                }
                with output_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(judgment_record,
                            ensure_ascii=False) + "\n")

                judged_pairs.add((query_id, candidate_id))

            elif judgment in {"q", "quit"}:
                return

            print("\033[2J\033[H", end="")


if __name__ == "__main__":

    file_path = Path("benchmarks/gold_v2/test_judgments.jsonl")
    queries_path = Path("benchmarks/gold_v2/queries.json")
    candidate_pool_path = Path("benchmarks/gold_v2/candidate_pools.json")
    corpus_path = Path("benchmarks/gold_v2/corpus.json")
    annotate(queries_path=queries_path, candidate_pool=candidate_pool_path,
             corpus_path=corpus_path, output_path=file_path)
