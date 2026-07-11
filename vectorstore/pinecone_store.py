import time
from typing import List

from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

from config import Config


class PineconeVectorStore:

    def __init__(self):

        self.pc = Pinecone(
            api_key=Config.PINECONE_API_KEY
        )

        self.index_name = Config.PINECONE_INDEX_NAME

        self._create_index()

        self.index = self.pc.Index(
            self.index_name
        )

    def _create_index(self):

        existing_indexes = [
            index.name
            for index in self.pc.list_indexes()
        ]

        if self.index_name in existing_indexes:
            return

        self.pc.create_index(
            name=self.index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    def upsert_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]]
    ):

        start = time.perf_counter()

        vectors = []

        for document, embedding in zip(
            documents,
            embeddings
        ):

            metadata = document.metadata.copy()

            metadata["text"] = document.page_content

            vectors.append(
                {
                    "id": metadata["chunk_id"],
                    "values": embedding,
                    "metadata": metadata,
                }
            )

        self.index.upsert(vectors=vectors)

        indexing_time = (
            time.perf_counter() - start
        )

        return indexing_time

    def query(
        self,
        query_embedding,
        top_k=5
    ):

        return self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

    def describe(self):

        return self.index.describe_index_stats()

    def delete_all(self):

        self.index.delete(delete_all=True)

# Singleton instance
pinecone_store = PineconeVectorStore()