from ingestion.ingest import IngestionPipeline
from retrieval.retriever import HybridRetriever


def main():

    # -----------------------------
    # Ingest Document
    # -----------------------------

    pipeline = IngestionPipeline()

    chunks, ingestion_metrics = pipeline.ingest(
        "data/sample.pdf"
    )

    print("\n========== INGESTION METRICS ==========\n")

    for key, value in ingestion_metrics.items():
        print(f"{key}: {value}")

    # -----------------------------
    # Retrieve
    # -----------------------------

    retriever = HybridRetriever()

    query = input("\nEnter your query: ")

    results, retrieval_metrics = retriever.retrieve(query)

    print("\n========== RETRIEVAL METRICS ==========\n")

    for key, value in retrieval_metrics.items():
        print(f"{key}: {value}")

    print("\n========== RETRIEVED CHUNKS ==========\n")

    for index, result in enumerate(results, start=1):

        print("=" * 100)

        print(f"Rank               : {index}")
        print(f"Source             : {result['source']}")
        print(f"Page               : {result['page']}")
        print(f"Dense Score        : {result['dense_score']:.4f}")
        print(f"BM25 Score         : {result['bm25_score']:.4f}")
        print(f"Hybrid Score       : {result['hybrid_score']:.4f}")
        print(f"Cosine Similarity  : {result['cosine_similarity']:.4f}")

        print("\nChunk:\n")
        print(result["text"][:600])

        print()


if __name__ == "__main__":
    main()