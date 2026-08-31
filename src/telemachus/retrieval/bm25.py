import bm25s

from telemachus.models import HFDatasetMetadata, ScoredDataset
from telemachus.representations.dataset import dataset_text


class BM25Retriever:
    def __init__(
        self,
        corpus: list[HFDatasetMetadata],
    ):
        self.corpus = corpus
        self.text_corpus = [dataset_text(dataset) for dataset in corpus]
        self.tokenizer = bm25s.tokenization.Tokenizer()
        self.corpus_tokens = self.tokenizer.tokenize(
            self.text_corpus,
            return_as="tuple",
        )
        self.bm25 = bm25s.BM25()
        self.bm25.index(self.corpus_tokens)

    def retrieve(
        self,
        query: str,
        k: int | None = None
    ) -> list[ScoredDataset]:
        if k is None:
            k = len(self.corpus)
        query_tokens = self.tokenizer.tokenize([query], update_vocab=False)
        bm25_return = self.bm25.retrieve(
            query_tokens=query_tokens, k=k, show_progress=False)

        document_indices = bm25_return.documents[0]
        scores = bm25_return.scores[0]

        scored_results = [
            ScoredDataset(
                dataset=self.corpus[int(document_index)],
                bm25_score=float(score),
            )
            for document_index, score in zip(document_indices, scores)
        ]

        return scored_results
