from ingestion.loader import DocumentLoader
from ingestion.splitter import DocumentSplitter


loader = DocumentLoader()

documents = loader.load("data/sample.pdf")

splitter = DocumentSplitter()

chunks = splitter.split_documents(documents)

print(f"\nTotal Chunks : {len(chunks)}\n")

for chunk in chunks[:5]:

    print("=" * 80)

    print(chunk.metadata)

    print(chunk.page_content[:250])

    print()