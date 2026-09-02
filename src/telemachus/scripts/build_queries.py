import json
from pathlib import Path

QUERIES = [
    # Paired queries

    # Lexical ↔ semantic
    "robot manipulation dataset",
    "demonstrations of robots learning to grasp and move physical objects",

    # Technical terminology ↔ plain English
    "binary sentiment classification dataset",
    "text labeled according to whether opinions are positive or negative",

    # Metadata-style ↔ user intent
    "question-answering dataset",
    "questions paired with answers for training a model to answer queries",

    # Short keyword ↔ conversational
    "legal document retrieval dataset",
    "I need a collection of legal documents that could be used to evaluate search and retrieval",

    # Hard constraints
    "English legal question-answering dataset",
    "English-language music audio dataset for genre classification",

    # Modality
    "medical images paired with questions and answers",
    "English speech recordings with transcriptions",

    # Ambiguity
    "dataset for financial reasoning",
    "dataset for studying climate change",

    # Multi-constraint / agent-friendly
    "English legal case documents suitable for evaluating semantic search",
    "English medical image question-answering data suitable for training a multimodal model",
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
    file_path = Path("benchmarks/gold_v2/queries.json")
    if not file_path.is_file() or input(f"The file exists. Overwrite {file_path}? (Y/N): ").strip().lower() in ["y", "yes"]:
        queries_to_json(file_path, QUERIES)
