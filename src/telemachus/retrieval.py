from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

from telemachus.models import HFDatasetMetadata

DEFAULT_EXPAND_FIELDS: list[str] = [
    "author",
    "cardData",
    "description",
    "downloads",
    "lastModified",
    "tags",
]

DEFAULT_MODEL: str = "sentence-transformers/all-MiniLM-L12-v2"


def get_hf_datasets(
    term: str | None = None,
    lim: int = 5,
    search_language: str = "en",
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


# Generated
def visualize_datasets(datasets: list[HFDatasetMetadata]) -> None:
    if not datasets:
        print("No datasets found.")
        return

    print(f"\n--- Found {len(datasets)} Datasets ---")
    for idx, item in enumerate(datasets, start=1):
        desc = item.description or "No description provided."
        clean_desc = desc[:300] + "..." if len(desc) > 300 else desc

        print(f"\n{idx}. [{item.id}] ({item.name})")
        print(
            f"   Author: {item.author} | Downloads: {item.downloads:,} | byte size: {item.byte_size} size category: {item.size_category}"
        )
        print(f"   Description: {clean_desc}")
        # print(f"Semantic Representation:\n{semantic_representation(item)}")


def semantic_representation(dataset: HFDatasetMetadata) -> str:
    name = dataset.name
    description = "Description: " + dataset.description if dataset.description else ""

    return f"Title: {name}\n{description}"


def cosine_similarity(query: np.ndarray, value: np.ndarray) -> float:

    return np.dot(query, value) / (norm(query) * norm(value))


# fix bonus


def rerank(
    scored_datasets: list[ScoredDataset],
    task_category: str | None,
    bonus: float = 0.08,
) -> list[ScoredDataset]:
    for ds in scored_datasets:
        if task_category in ds.dataset.task_categories:
            ds.final_score = ds.semantic_score + bonus
        else:
            ds.final_score = ds.semantic_score

    return scored_datasets


@dataclass
class ScoredDataset:
    dataset: HFDatasetMetadata
    semantic_score: float
    final_score: float


@dataclass
class EvaluationCase:
    query: str
    relevant: set[str]
    task_category: str | None


def main() -> None:

    # Dummy labels for query, as determined by me.
    # Query: “dataset for training robots to perform physical manipulation tasks”
    eval_cases = [
        EvaluationCase(
            query="dataset for training robots to perform physical manipulation tasks",
            relevant={
                "genrobot2025/10Kh-RealOmin-OpenData",
                "XDOF/ABC-130k",
                "InternRobotics/InternData-A1",
                "cadene/droid",
            },
            task_category="robotics",
        ),
        EvaluationCase(
            query="English dataset for training a text classification model",
            relevant={
                "nyu-mll/glue",
                "aps/super_glue",
                "stanfordnlp/imdb",
            },
            task_category="text-classification",
        ),
        EvaluationCase(
            query="English dataset for training a question-answering model",
            relevant={
                "allenai/ai2_arc",
                "rajpurkar/squad",
            },
            task_category="question-answering",
        ),
        EvaluationCase(
            query="English legal text corpus suitable for training or fine-tuning a language model",
            relevant={
                "pile-of-law/pile-of-law",
                "mratanusarkar/Indian-Laws",
                "a2aj/canadian-case-law",
                "HFforLegal/case-law",
            },
            task_category=None,
        ),
        EvaluationCase(
            query="English medical text dataset suitable for training or fine-tuning a language model",
            relevant={
                "lavita/medical-qa-datasets",
                "medalpaca/medical_meadow_medqa",
                "medalpaca/medical_meadow_medical_flashcards",
                "medalpaca/medical_meadow_wikidoc",
                "FreedomIntelligence/medical-o1-reasoning-SFT",
            },
            task_category=None,
        ),
        EvaluationCase(
            query="English financial text corpus suitable for training or fine-tuning a language model",
            relevant={
                "artefactory/Argimi-Ardian-Finance-10k-text",
                "vidore/vidore_v3_finance_en",
                "gbharti/finance-alpaca",
            },
            task_category=None,
        ),
    ]
    # 1. Fetch datasets
    results_by_category = ["robotics", "text-classification", "question-answering"]
    results_by_term = ["law", "medical", "finance"]

    raw_corpus: list[HFDatasetMetadata] = []

    for topic in results_by_category:
        raw_corpus.extend(
            get_hf_datasets(task_category=topic, lim=5, search_language="en")
        )
    for topic in results_by_term:
        raw_corpus.extend(get_hf_datasets(term=topic, lim=5, search_language="en"))

    seen_ids = set()
    corpus: list[HFDatasetMetadata] = []
    for ds in raw_corpus:
        if ds.id not in seen_ids:
            seen_ids.add(ds.id)
            corpus.append(ds)

    print(f"Loaded {len(corpus)} unique datasets into evaluation corpus.\n")

    # 2. Make semantic text
    semantic_ds = [semantic_representation(ds) for ds in corpus]

    # 3. Embed datasets
    model = SentenceTransformer(DEFAULT_MODEL)
    embeddings_to_compare = model.encode(
        semantic_ds,
        batch_size=32,
        show_progress_bar=len(semantic_ds) > 50,
        convert_to_numpy=True,
    )

    reciprocal_ranks: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []

    benchmark_output = {
        "model": DEFAULT_MODEL,
        "top_k": 5,
        "queries": [],
    }

    for case in eval_cases:
        query_embedding = model.encode(case.query)

        # Score embeddings against corpus
        # fix referencing 'corpus' instead of undefined 'all_ds'
        scored_results = [
            ScoredDataset(
                dataset=ds,
                semantic_score=float(cosine_similarity(query_embedding, embedded_ds)),
                final_score=0,
            )
            for ds, embedded_ds in zip(corpus, embeddings_to_compare)
        ]
        # pre_reranked_scored_results.sort(key=lambda item: item.score, reverse=True)

        scored_results = rerank(scored_results, case.task_category)
        scored_results.sort(key=lambda item: item.final_score, reverse=True)

        # For calculating RR
        first_relevant_rank: int | None = None

        for rank, res in enumerate(scored_results, start=1):
            if res.dataset.id in case.relevant:
                first_relevant_rank = rank
                break
        rr = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0
        reciprocal_ranks.append(rr)

        # Top k, add global variable/make dependent on number of labeled examples.
        top_k = 5
        top_results = scored_results[:top_k]

        # Calculate Precision@K
        hits = sum(1 for res in top_results if res.dataset.id in case.relevant)
        precision_at_k = hits / top_k
        precision_scores.append(precision_at_k)

        # Calculate Recall@K
        # Change this
        recall_at_k = hits / len(case.relevant)
        recall_scores.append(recall_at_k)

        # Print query benchmark breakdown
        print(f'\nQuery: "{case.query}"')
        print(f"Precision@{top_k}: {hits}/{top_k} ({precision_at_k:.0%})")
        print(f"Recall@{top_k}: {hits}/{len(case.relevant)} ({recall_at_k:.0%})")
        print("-" * 70)

        for rank, res in enumerate(top_results, start=1):
            is_relevant = (
                "RELEVANT" if res.dataset.id in case.relevant else "NOT RELEVANT"
            )
            print(
                f"{rank}. [{res.final_score:.4f}] {res.dataset.id:<45} Relevant: {is_relevant}"
            )
        benchmark_output["queries"].append(
            {
                "query": case.query,
                "relevant_ids": sorted(case.relevant),
                "precision_at_5": precision_at_k,
                "recall_at_5": recall_at_k,
                "reciprocal_rank": rr,
                "top_results": [
                    {
                        "rank": rank,
                        "dataset_id": result.dataset.id,
                        "score": result.final_score,
                        "relevant": result.dataset.id in case.relevant,
                    }
                    for rank, result in enumerate(top_results, start=1)
                ],
            }
        )
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    mean_p5 = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    mean_r5 = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    benchmark_output["summary"] = {
        "num_queries": len(eval_cases),
        "mean_precision_at_5": mean_p5,
        "mean_reciprocal_rank": mrr,
        "mean_recall": mean_r5,
    }

    print("\n" + "=" * 70)
    print("GLOBAL BENCHMARK EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Test Queries:     {len(eval_cases)}")
    print(f"Mean Reciprocal Rank:   {mrr:.4f}")
    print(f"Mean Precision@5:       {mean_p5:.1%}")
    print(f"Mean Recall@5:       {mean_r5:.1%}")

    print("=" * 70)

    output_path = Path("results/reranked_semantic.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(benchmark_output, f, indent=2, ensure_ascii=False)


# Test / visualization
if __name__ == "__main__":
    main()
