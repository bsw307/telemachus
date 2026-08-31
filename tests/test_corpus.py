from pathlib import Path

from telemachus.models import HFDatasetMetadata
from telemachus.sources.corpus import corpus_from_json, corpus_to_json


def test_corpus_json_round_trip(tmp_path: Path) -> None:
    corpus = [
        HFDatasetMetadata(
            id="example/test-dataset",
            name="Test Dataset",
            author="example",
            description="A small test dataset.",
        )
    ]

    output_path = tmp_path / "corpus.json"

    corpus_to_json(
        corpus=corpus,
        output_path=output_path,
        metadata={
            "version": "test",
            "source": "huggingface",
        },
    )

    loaded = corpus_from_json(output_path)

    assert loaded == corpus
