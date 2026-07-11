from ingestion.ingest import IngestionPipeline

pipeline = IngestionPipeline()

chunks, metrics = pipeline.ingest(
    "data/sample.pdf"
)

print()

print(metrics)