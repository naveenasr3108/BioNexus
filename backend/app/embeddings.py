from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:

    def __init__(self):
        print("Loading BGE-M3 model...")
        self.model = SentenceTransformer("BAAI/bge-m3")
        print("BGE-M3 model loaded")

    def generate(self, chunks: list[str]):
        """Generate embeddings for paper chunks."""
        return self.model.encode(
            chunks,
            normalize_embeddings=True
        )

    def embed_query(self, query: str):
        """Generate an embedding for a user's question."""
        return self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]