from types import SimpleNamespace

from telemachus.models import HFDatasetMetadata, _clean_str


def test_clean_str_collapses_whitespace():
    text = "Hello \n\n   world\tfrom   Telemachus"

    assert _clean_str(text) == "Hello world from Telemachus"


def test_clean_str_handles_invalid_values():
    assert _clean_str(None) == ""
    assert _clean_str("") == ""
    assert _clean_str(123) == ""


def test_metadata_defaults():
    metadata = HFDatasetMetadata(
        id="example/test",
        name="test",
        author="example",
    )

    assert metadata.license == []
    assert metadata.language == []
    assert metadata.task_categories == []
    assert metadata.tags == []
    assert metadata.downloads == 0
    assert metadata.byte_size is None
    assert metadata.last_updated is None


def test_metadata_list_defaults_are_independent():
    first = HFDatasetMetadata(
        id="example/first",
        name="first",
        author="example",
    )

    second = HFDatasetMetadata(
        id="example/second",
        name="second",
        author="example",
    )

    first.tags.append("test")

    assert second.tags == []


def test_from_hf_api_parses_dataset_metadata():
    card = SimpleNamespace(
        pretty_name="Example Dataset",
        language=["en", "sv"],
        license="mit",
        task_categories=["text-classification"],
        dataset_info={
            "download_size": 1_000_000,
        },
    )

    ds = SimpleNamespace(
        id="example/example-dataset",
        author="example",
        card_data=card,
        description="An   example\n dataset.",
        downloads=5000,
        last_modified=None,
        tags=["nlp", "classification"],
        size_categories=None,
    )

    metadata = HFDatasetMetadata.from_hf_api(ds)

    assert metadata.id == "example/example-dataset"
    assert metadata.name == "Example Dataset"
    assert metadata.author == "example"

    assert metadata.language == ["en", "sv"]
    assert metadata.license == ["mit"]
    assert metadata.task_categories == ["text-classification"]

    assert metadata.description == "An example dataset."
    assert metadata.downloads == 5000
    assert metadata.byte_size == 1_000_000
    assert metadata.tags == ["nlp", "classification"]


def test_from_hf_api_converts_single_values_to_lists():
    card = SimpleNamespace(
        pretty_name=None,
        language="en",
        license="apache-2.0",
        task_categories="question-answering",
        dataset_info=None,
    )

    ds = SimpleNamespace(
        id="example/my-dataset",
        author=None,
        card_data=card,
        description=None,
        downloads=None,
        last_modified=None,
        tags=None,
        size_categories=None,
    )

    metadata = HFDatasetMetadata.from_hf_api(ds)

    assert metadata.language == ["en"]
    assert metadata.license == ["apache-2.0"]
    assert metadata.task_categories == ["question-answering"]


def test_from_hf_api_handles_missing_card_data():
    ds = SimpleNamespace(
        id="example/my-dataset",
        author=None,
        card_data=None,
        description=None,
        downloads=None,
        last_modified=None,
        tags=None,
        size_categories=None,
    )

    metadata = HFDatasetMetadata.from_hf_api(ds)

    assert metadata.author == "example"
    assert metadata.name == "my-dataset"

    assert metadata.language == []
    assert metadata.license == []
    assert metadata.task_categories == []
    assert metadata.tags == []
    assert metadata.description == ""
    assert metadata.byte_size is None


def test_from_hf_api_uses_canonical_author_without_namespace():
    ds = SimpleNamespace(
        id="squad",
        author=None,
        card_data=None,
        description=None,
        downloads=None,
        last_modified=None,
        tags=None,
        size_categories=None,
    )

    metadata = HFDatasetMetadata.from_hf_api(ds)

    assert metadata.author == "canonical"
    assert metadata.name == "squad"
