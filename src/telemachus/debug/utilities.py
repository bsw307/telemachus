from telemachus.models import HFDatasetMetadata


def visualize_datasets(datasets: list[HFDatasetMetadata]) -> None:

    if not datasets:
        print("No datasets found.")
    else:
        print(f"\n--- Found {len(datasets)} Datasets ---")
        for idx, item in enumerate(datasets, start=1):
            desc = item.description or "No description provided."
            clean_desc = desc[:300] + "..." if len(desc) > 300 else desc

            print(f"\n{idx}. [{item.id}] ({item.name})")
            print(
                f"   Author: {item.author} | Downloads: {item.downloads:,} | byte size: {item.byte_size} size category: {item.size_category}"
            )
            print(f"   Description: {clean_desc}")
