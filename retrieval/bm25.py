from typing import List

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self):
        self.bm25 = None
        self.documents: List[Document] = []

    def build(self, documents: List[Document]) -> None:
        """
        Build the BM25 index from document chunks.
        """

        self.documents = documents

        corpus = [
            document.page_content.lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 5):

        if self.bm25 is None:
            raise ValueError(
                "BM25 index has not been built. Run the ingestion pipeline first."
            )

        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(query_tokens)

        ranked_results = sorted(
            zip(self.documents, scores),
            key=lambda item: item[1],
            reverse=True
        )

        return ranked_results[:top_k]

    def clear(self) -> None:
        """
        Clear the BM25 index.
        """

        self.bm25 = None
        self.documents = []

    def size(self) -> int:
        """
        Number of indexed chunks.
        """

        return len(self.documents)


# Singleton instance
bm25_retriever = BM25Retriever()