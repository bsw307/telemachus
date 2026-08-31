from telemachus.models import HFDatasetMetadata


def dataset_text(dataset: HFDatasetMetadata) -> str:
    name = dataset.name
    description = "Description: " + dataset.description if dataset.description else ""

    return f"Title: {name}\n{description}"
