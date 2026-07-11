import time

from sklearn.metrics.pairwise import cosine_similarity

from config import Config

from ingestion.embedder import EmbeddingEngine

from cache.embedding_cache import embedding_cache
from retrieval.bm25 import bm25_retriever
from retrieval.hybrid import HybridSearch
from vectorstore.pinecone_store import pinecone_store


class HybridRetriever:

    def __init__(self):

        self.embedder = EmbeddingEngine()

    def retrieve(self, query: str):

        start = time.perf_counter()

        query_embedding = self.embedder.embed_query(query)

        dense_response = pinecone_store.query(
            query_embedding=query_embedding,
            top_k=Config.TOP_K
        )

        sparse_response = bm25_retriever.search(
            query=query,
            top_k=Config.TOP_K
        )

        merged = {}

        # ----------------------------
        # Dense Results
        # ----------------------------

        for match in dense_response.matches:

            chunk_id = match.id

            merged[chunk_id] = {

                "chunk_id": chunk_id,

                "text": match.metadata["text"],

                "source": match.metadata.get("source"),

                "page": match.metadata.get("page"),

                "dense_score": match.score,

                "bm25_score": 0.0,

                "hybrid_score": 0.0,

                "cosine_similarity": 0.0

            }

        # ----------------------------
        # Sparse Results
        # ----------------------------

        for document, score in sparse_response:

            chunk_id = document.metadata["chunk_id"]

            if chunk_id not in merged:

                merged[chunk_id] = {

                    "chunk_id": chunk_id,

                    "text": document.page_content,

                    "source": document.metadata.get("source"),

                    "page": document.metadata.get("page"),

                    "dense_score": 0.0,

                    "bm25_score": score,

                    "hybrid_score": 0.0,

                    "cosine_similarity": 0.0

                }

            else:

                merged[chunk_id]["bm25_score"] = score

        # ----------------------------
        # Hybrid Score Fusion
        # ----------------------------

        results = HybridSearch.fuse(merged)

        # ----------------------------
        # Cosine Similarity
        # ----------------------------

        for item in results:

            chunk_embedding = embedding_cache.get(
                item["chunk_id"]
            )

            if chunk_embedding is None:
                continue

            item["cosine_similarity"] = cosine_similarity(
                [query_embedding],
                [chunk_embedding]
            )[0][0]

        # ----------------------------
        # Top K Results
        # ----------------------------

        results = results[:Config.TOP_K]

        retrieval_time = (
            time.perf_counter() - start
        )

        if results:

            metrics = {

                "retrieval_time": retrieval_time,

                "retrieved_chunks": len(results),

                "average_dense_score": sum(
                    r["dense_score"]
                    for r in results
                ) / len(results),

                "average_bm25_score": sum(
                    r["bm25_score"]
                    for r in results
                ) / len(results),

                "average_hybrid_score": sum(
                    r["hybrid_score"]
                    for r in results
                ) / len(results),

                "max_cosine_similarity": max(
                    r["cosine_similarity"]
                    for r in results
                ),

                "average_cosine_similarity": sum(
                    r["cosine_similarity"]
                    for r in results
                ) / len(results)

            }

        else:

            metrics = {

                "retrieval_time": retrieval_time,

                "retrieved_chunks": 0,

                "average_dense_score": 0.0,

                "average_bm25_score": 0.0,

                "average_hybrid_score": 0.0,

                "max_cosine_similarity": 0.0,

                "average_cosine_similarity": 0.0

            }

        return results, metrics