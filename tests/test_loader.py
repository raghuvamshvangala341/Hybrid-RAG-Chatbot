from ingestion.loader import DocumentLoader


loader = DocumentLoader()

documents = loader.load("data/sample.pdf")

print(f"\nLoaded {len(documents)} documents\n")

for document in documents:

    print("=" * 80)

    print(document.metadata)

    print(document.page_content[:300])