import os
from dotenv import load_dotenv

load_dotenv()


class Config:

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5"
    )

    LLM_MODEL = os.getenv(
        "LLM_MODEL",
        "llama-3.3-70b-versatile"
    )

    CHUNK_SIZE = int(
        os.getenv("CHUNK_SIZE", 1000)
    )

    CHUNK_OVERLAP = int(
        os.getenv("CHUNK_OVERLAP", 200)
    )

    TOP_K = int(
        os.getenv("TOP_K", 5)
    )

    HYBRID_ALPHA = float(
        os.getenv("HYBRID_ALPHA", 0.7)
    )