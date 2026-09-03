import json
from pathlib import Path


def load_judgments(
    judgment_path: Path
) -> dict[tuple[str, str], int]:

    result: dict[tuple[str, str], int] = {}
    with judgment_path.open("r", encoding="utf-8") as f:
        for line in f:
            row_dict = json.loads(line)
            query_id = row_dict["query_id"]
            dataset_id = row_dict["dataset_id"]
            relevance = row_dict["relevance"]
            result[(query_id, dataset_id)] = relevance

    return result
