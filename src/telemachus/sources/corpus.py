import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from telemachus.models import HFDatasetMetadata
from telemachus.sources.huggingface import get_hf_datasets


def generate_hf_corpus(
    search_terms: list[str] | None = None,
    search_count: int = 20,
    task_categories: list[str] | None = None,
    task_count: int = 20,
    search_language: str | None = "en",
) -> list[HFDatasetMetadata]:

    raw_corpus: list[HFDatasetMetadata] = []

    for term in search_terms or []:
        raw_corpus.extend(
            get_hf_datasets(
                term=term,
                lim=search_count,
                search_language=search_language
            )
        )
    for task in task_categories or []:
        raw_corpus.extend(
            get_hf_datasets(
                task_category=task,
                lim=task_count,
                search_language=search_language
            )
        )

    seen_ids: set[str] = set()
    corpus: list[HFDatasetMetadata] = []
    for ds in raw_corpus:
        if ds.id not in seen_ids:
            seen_ids.add(ds.id)
            corpus.append(ds)
    return corpus


def _json_serializer(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def corpus_to_json(
    corpus: list[HFDatasetMetadata],
    output_path: Path,
    *,
    metadata: dict[str, object],
) -> None:

    payload = {
        "metadata": metadata,
        "datasets": [
            asdict(dataset)
            for dataset in corpus
        ],
    }
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
            default=_json_serializer,
        )


def corpus_from_json(
    input_path: Path,
) -> list[HFDatasetMetadata]:

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

        for dataset in payload["datasets"]:
            if dataset["last_updated"] is not None:
                dataset["last_updated"] = datetime.fromisoformat(
                    dataset["last_updated"]
                )

    return [
        HFDatasetMetadata(**dataset)
        for dataset in payload["datasets"]
    ]
