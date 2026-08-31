from telemachus.models import HFDatasetMetadata
from telemachus.retrieval.bm25 import BM25Retriever

corpus = [
    HFDatasetMetadata(
        id="robotics/manipulation",
        name="Robot Manipulation Dataset",
        author="test",
        description="Robot arm demonstrations for physical manipulation tasks.",
    ),
    HFDatasetMetadata(
        id="nlp/classification",
        name="Text Classification Dataset",
        author="test",
        description="English text classification examples.",
    ),
    HFDatasetMetadata(
        id="finance/corpus",
        name="Financial Text Corpus",
        author="test",
        description="Financial reports and market documents.",
    ),
]

retriever = BM25Retriever(corpus)

results = retriever.retrieve(
    "robot physical manipulation",
    k=3,
)

for rank, result in enumerate(results, start=1):
    print(
        rank,
        result.dataset.id,
        result.bm25_score,
    )


def test_bm25_retrieves_lexically_relevant_dataset_first():
    retriever = BM25Retriever(corpus)

    results = retriever.retrieve(
        "robot physical manipulation",
        k=3,
    )

    assert results[0].dataset.id == "robotics/manipulation"
    assert results[0].bm25_score > 0
    assert results[1].bm25_score == 0
    assert results[2].bm25_score == 0


def test_bm25_none_k_returns_full_corpus():
    retriever = BM25Retriever(corpus)

    results = retriever.retrieve("robot", k=None)

    assert len(results) == len(corpus)
