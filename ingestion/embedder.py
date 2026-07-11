import time
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import Config


class EmbeddingEngine:

    def __init__(self):

        self.model = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32
            }
        )

    def embed_documents(
        self,
        documents: List[Document]
    ):

        start_time = time.perf_counter()

        texts = [
            document.page_content
            for document in documents
        ]

        embeddings = self.model.embed_documents(texts)

        embedding_time = time.perf_counter() - start_time

        return embeddings, embedding_time

    def embed_query(self, query: str):

        return self.model.embed_query(query)