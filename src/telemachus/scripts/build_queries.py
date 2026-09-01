import json
from pathlib import Path

QUERIES = [
    "English medical question-answering dataset suitable for fine-tuning a language model",
    "Test1",
    "Test2",
]


def queries_to_json(
    output_path: Path,
    queries: list[str],
) -> None:

    payload = [
        {
            "id": f"q{i:03}",
            "query": query,
        }
        for i, query in enumerate(queries, start=1)
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
            ensure_ascii=False,
        )


if __name__ == "__main__":
    file_path = Path("benchmarks/gold_v2/test_queries.json")
    if not file_path.is_file() or input(f"The file exists. Overwrite {file_path}? (Y/N): ").strip().lower() in ["y", "yes"]:
        queries_to_json(file_path, QUERIES)
