from pathlib import Path

from telemachus.sources.corpus import corpus_to_json, generate_hf_corpus

SEARCH_TERMS = []
TASK_CATEGORIES = []


def build_corpus(
    output_path: Path,
    *,
    search_terms: list[str],
    task_categories: list[str],
    search_count: int,
    task_count: int,
    version: str,
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
    )

    metadata = {
        "version": version,
        "source": "huggingface",
        "search_terms": search_terms,
        "task_categories": task_categories,
        "search_count": search_count,
        "task_count": task_count,
        "num_datasets": len(corpus),
    }
    corpus_to_json(
        corpus=corpus,
        output_path=output_path,
        metadata=metadata,
    )


if __name__ == "__main__":
    build_corpus(
        output_path=Path("evaluation/data/gold_v2/corpus.json"),
        search_terms=SEARCH_TERMS,
        task_categories=TASK_CATEGORIES,
        search_count=25,
        task_count=25,
        version="gold-v2",
    )
