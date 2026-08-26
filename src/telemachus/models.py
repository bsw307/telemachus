from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# Card Data tuple to be searched/embedded


def _clean_str(text: Any) -> str:
    """Safely converts input to string and collapses all whitespace into single spaces."""
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class HFDatasetMetadata:
    # Unchanging metrics
    id: str
    name: str
    author: str
    license: list[str] = field(default_factory=list)
    description: str | None = None

    # Quantitative metrics (Nullable as API often omits them)
    downloads: int | None = 0
    byte_size: int | None = None
    size_category: int | None = None

    # Categories & Tags
    language: list[str] = field(default_factory=list)
    task_categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # Timestamps
    last_updated: datetime | None = None

    @classmethod
    def from_hf_api(cls, ds) -> HFDatasetMetadata:

        # 1. ID & Author (Top-level DatasetInfo)
        ds_id = getattr(ds, "id", "")
        ds_author = getattr(ds, "author", None) or (
            ds_id.split("/")[0] if "/" in ds_id else "canonical"
        )

        # 2. CardData attributes (Direct object property lookup)
        card = getattr(ds, "card_data", None)
        pretty_name = getattr(card, "pretty_name", None) if card else None
        ds_name = pretty_name or (ds_id.split("/")[-1] if ds_id else "")

        raw_lang = getattr(card, "language", None) if card else None
        if isinstance(raw_lang, str):
            ds_languages = [raw_lang]
        elif isinstance(raw_lang, list):
            ds_languages = raw_lang
        else:
            ds_languages = []

        raw_license = getattr(card, "license", None) if card else None
        if isinstance(raw_license, str):
            ds_license = [raw_license]
        elif isinstance(raw_license, list):
            ds_license = raw_license
        else:
            ds_license = []
        raw_tasks = getattr(card, "task_categories", None) if card else None
        ds_task_categories = (
            raw_tasks
            if isinstance(raw_tasks, list)
            else ([raw_tasks] if raw_tasks else [])
        )

        # 3. Nested size specs inside card.dataset_info
        ds_byte_size = None
        info_spec = getattr(card, "dataset_info", None) if card else None

        if isinstance(info_spec, dict):
            ds_byte_size = info_spec.get("download_size")
        elif info_spec is not None:
            ds_byte_size = getattr(info_spec, "download_size", None)

        ds_size_category = getattr(ds, "size_categories", None)
        # 4. Top-level DatasetInfo attributes
        raw_description = getattr(ds, "description", None)
        ds_description = _clean_str(raw_description)

        ds_downloads = getattr(ds, "downloads", None)
        ds_last_updated = getattr(ds, "last_modified", None)
        ds_tags = getattr(ds, "tags", []) or []

        return cls(
            id=ds_id,
            name=ds_name,
            author=ds_author,
            license=ds_license,
            downloads=ds_downloads,
            byte_size=ds_byte_size,
            size_category=ds_size_category,
            last_updated=ds_last_updated,
            description=ds_description,
            task_categories=ds_task_categories,
            language=ds_languages,
            tags=ds_tags,
        )


# Tests

if __name__ == "__main__":
    print("🚀 Running local dataclass validation tests...")
    # 1. Create a dummy instance matching the dataclass schema
    test_metadata = HFDatasetMetadata(
        id="meta-llama/Llama-3-8B",
        name="Llama-3-8B",
        author="meta-llama",
        license="llama3",
        downloads=1500000,
        byte_size=16000000000,
        last_updated=datetime.now(UTC),
        description="A large language model.",
        language=["en"],
        task_categories=["text-generation"],
        tags=["llm", "pretrained"],
    )

    # 2. Run Assertions
    assert test_metadata.id == "meta-llama/Llama-3-8B", "ID assignment failed"
    assert "llm" in test_metadata.tags, "Tag array assignment failed"
    assert test_metadata.language == ["en"], "Language parsing failed"

    # 3. Test default factories and optional fields
    empty_metadata = HFDatasetMetadata(
        id="empty/test",
        name="test",
        author="test",
    )
    assert len(empty_metadata.tags) == 0, (
        "Default factory failed to initialize empty list"
    )
    assert empty_metadata.downloads == 0, "Default downloads fallback failed"

    print("✅ All local tests passed successfully!")
