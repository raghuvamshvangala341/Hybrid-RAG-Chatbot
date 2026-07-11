from ingestion.loader import DocumentLoader
from ingestion.splitter import DocumentSplitter
from ingestion.embedder import EmbeddingEngine


loader = DocumentLoader()

documents = loader.load("data/sample.pdf")

splitter = DocumentSplitter()

chunks = splitter.split_documents(documents)

embedder = EmbeddingEngine()

embeddings, embedding_time = embedder.embed_documents(chunks)

print()

print(f"Chunks : {len(chunks)}")

print(f"Embedding Dimension : {len(embeddings[0])}")

print(f"Embedding Time : {embedding_time:.3f} sec")