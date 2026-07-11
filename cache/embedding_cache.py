class EmbeddingCache:
    """
    Stores chunk embeddings in memory.

    Key:
        chunk_id

    Value:
        embedding vector
    """

    def __init__(self):
        self.embeddings = {}

    def add(self, chunk_id: str, embedding: list[float]) -> None:
        self.embeddings[chunk_id] = embedding

    def get(self, chunk_id: str):
        return self.embeddings.get(chunk_id)

    def clear(self) -> None:
        self.embeddings.clear()

    def size(self) -> int:
        return len(self.embeddings)


# Singleton instance
embedding_cache = EmbeddingCache()