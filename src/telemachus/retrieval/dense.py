import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

from telemachus.models import HFDatasetMetadata, ScoredDataset
from telemachus.representations.dataset import dataset_text

DEFAULT_MODEL: str = "sentence-transformers/all-MiniLM-L12-v2"


def cosine_similarity(query: np.ndarray, value: np.ndarray) -> float:

    return np.dot(query, value) / (norm(query) * norm(value))


class DenseRetriever:
    def __init__(
        self,
        corpus: list[HFDatasetMetadata],
        model_name: str = DEFAULT_MODEL
    ):
        self.corpus = corpus
        self.model_name = model_name
        self.text_corpus = [
            dataset_text(dataset)
            for dataset in corpus
        ]
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            self.text_corpus,
            batch_size=32,
            show_progress_bar=len(self.corpus) > 50,
            convert_to_numpy=True
        )

    def retrieve(
        self,
        query: str,
        k: int | None = None
    ) -> list[ScoredDataset]:

        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )
        scored_results = [
            ScoredDataset(
                dataset=ds,
                dense_score=float(
                    cosine_similarity(query_embedding, embedded_ds)
                )
            )
            for ds, embedded_ds in zip(self.corpus, self.embeddings)
        ]
        scored_results.sort(key=lambda item: item.dense_score, reverse=True)

        if k is not None:
            scored_results = scored_results[:k]

        return scored_results
