from ingestion.ingest import IngestionPipeline

from llm.rag_chain import RAGChain


pipeline = IngestionPipeline()

pipeline.ingest(
    "data/sample.pdf"
)

rag = RAGChain()

question = input("Question: ")

response = rag.ask(question)

print()

print("=" * 100)

print("ANSWER")

print("=" * 100)

print(response["answer"])

print()

print("=" * 100)

print("SOURCES")

print("=" * 100)

for source in response["sources"]:

    print(source)

print()

print("=" * 100)

print("METRICS")

print("=" * 100)

for key, value in response["metrics"].items():

    print(f"{key}: {value}")