from ingestion.loader import DocumentLoader
from ingestion.splitter import DocumentSplitter
from ingestion.embedder import EmbeddingEngine

from vectorstore.pinecone_store import PineconeVectorStore


loader = DocumentLoader()

documents = loader.load("data/sample.pdf")

splitter = DocumentSplitter()

chunks = splitter.split_documents(documents)

embedder = EmbeddingEngine()

embeddings, embedding_time = embedder.embed_documents(chunks)

store = PineconeVectorStore()

indexing_time = store.upsert_documents(
    chunks,
    embeddings
)

print()

print("Embedding Time :", embedding_time)

print("Indexing Time :", indexing_time)

print()

print(store.describe_index())