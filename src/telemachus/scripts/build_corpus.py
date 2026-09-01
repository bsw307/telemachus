from pathlib import Path

from telemachus.sources.corpus import corpus_to_json, generate_hf_corpus

SEARCH_TERMS = [
    "medical",
    "legal",
    "finance",
    "biology",
    "chemistry",
    "climate",
    "code",
    "cybersecurity",
    "geospatial",
    "music",
]
TASK_CATEGORIES = [
    "text-classification",
    "question-answering",
    "summarization",
    "text-generation",
    "sentence-similarity",
    "text-retrieval",
    "image-classification",
    "object-detection",
    "visual-question-answering",
    "automatic-speech-recognition",
    "time-series-forecasting",
    "robotics",
]


def build_corpus(
    output_path: Path,
    *,
    search_terms: list[str],
    task_categories: list[str],
    search_count: int,
    task_count: int,
    version: str,
    search_language: str | None = "en"
) -> None:

    if not search_terms and not task_categories:
        raise ValueError(
            "At least one search term or task category is required."
        )

    corpus = generate_hf_corpus(
        search_terms=search_terms,
        search_count=search_count,
        task_categories=task_categories,
        task_count=task_count,
        search_language=search_language
    )

    metadata = {
        "version": version,
        "source": "huggingface",
        "search_terms": search_terms,
        "task_categories": task_categories,
        "search_count": search_count,
        "task_count": task_count,
        "search_language": search_language,
        "num_datasets": len(corpus),
    }
    corpus_to_json(
        corpus=corpus,
        output_path=output_path,
        metadata=metadata,
    )


if __name__ == "__main__":
    file_path = Path("benchmarks/gold_v2/corpus.json")
    if not file_path.is_file() or input(f"The file exists. Overwrite {file_path}? (Y/N): ").strip().lower() in ["y", "yes"]:
        build_corpus(
            output_path=file_path,
            search_terms=SEARCH_TERMS,
            task_categories=TASK_CATEGORIES,
            search_language=None,
            search_count=20,
            task_count=25,
            version="gold-v2",
        )
