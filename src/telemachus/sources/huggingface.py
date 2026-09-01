from collections.abc import Sequence

from huggingface_hub import HfApi

from telemachus.models import HFDatasetMetadata

DEFAULT_EXPAND_FIELDS: list[str] = [
    "author",
    "cardData",
    "description",
    "downloads",
    "lastModified",
    "tags",
]


def get_hf_datasets(
    term: str | None = None,
    lim: int = 5,
    search_language: str | None = "en",
    task_category: str | None = None,
    extra_filters: list[str] | None = None,
    sort_by: str = "downloads",
    expand: Sequence[str] | None = None,
    api: HfApi | None = None,
) -> list[HFDatasetMetadata]:

    client = api or HfApi()
    fields = list(expand) if expand is not None else DEFAULT_EXPAND_FIELDS
    filters: list[str] = []

    if search_language:
        filters.append(f"language:{search_language}")
    if task_category:
        filters.append(f"task_categories:{task_category}")
    if extra_filters:
        filters.extend(extra_filters)

    hf_results = client.list_datasets(
        search=term,
        filter=filters if filters else None,
        limit=lim,
        sort=sort_by,
        direction=-1,
        expand=fields,
    )

    return [HFDatasetMetadata.from_hf_api(raw) for raw in hf_results]
