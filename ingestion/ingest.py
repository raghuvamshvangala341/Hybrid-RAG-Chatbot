import time
from pathlib import Path

from ingestion.loader import DocumentLoader
from ingestion.splitter import DocumentSplitter
from ingestion.embedder import EmbeddingEngine

from cache.embedding_cache import embedding_cache
from retrieval.bm25 import bm25_retriever
from vectorstore.pinecone_store import pinecone_store


class IngestionPipeline:

    def __init__(self):

        self.loader = DocumentLoader()

        self.splitter = DocumentSplitter()

        self.embedder = EmbeddingEngine()

    def ingest(self, file_path: str | Path):

        total_start = time.perf_counter()

        # ---------------------------------
        # Load Documents
        # ---------------------------------

        documents = self.loader.load(file_path)

        # ---------------------------------
        # Split Documents
        # ---------------------------------

        chunks = self.splitter.split_documents(
            documents
        )

        # ---------------------------------
        # Generate Embeddings
        # ---------------------------------

        embeddings, embedding_time = (
            self.embedder.embed_documents(chunks)
        )

        # ---------------------------------
        # Refresh Embedding Cache
        # ---------------------------------

        embedding_cache.clear()

        for chunk, embedding in zip(chunks, embeddings):

            embedding_cache.add(
                chunk.metadata["chunk_id"],
                embedding
            )

        # ---------------------------------
        # Refresh BM25 Index
        # ---------------------------------

        bm25_retriever.clear()

        bm25_retriever.build(chunks)

        # ---------------------------------
        # Refresh Pinecone Index
        # ---------------------------------

        pinecone_store.delete_all()

        indexing_time = (
            pinecone_store.upsert_documents(
                chunks,
                embeddings
            )
        )

        total_time = (
            time.perf_counter() - total_start
        )

        metrics = {

            "documents": len(documents),

            "chunks": len(chunks),

            "embedding_time": embedding_time,

            "indexing_time": indexing_time,

            "total_ingestion_time": total_time,

            "embedding_cache_size":
                embedding_cache.size(),

            "bm25_index_size":
                bm25_retriever.size()

        }

        return chunks, metrics