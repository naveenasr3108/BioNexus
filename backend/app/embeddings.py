from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")

    def generate(self, chunks: list[str]):
        vectors = self.model.encode(chunks, normalize_embeddings=True)
        return vectors  # shape: (num_chunks, 1024)