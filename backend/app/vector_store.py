from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid


class VectorStore:

    def __init__(self):
        self.client = QdrantClient(path="qdrant_data")
        self.collection = "bionexus_papers"

        collections = self.client.get_collections().collections

        if self.collection not in [c.name for c in collections]:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=1024,
                    distance=Distance.COSINE
                )
            )

    def add_chunks(self, file_id, title, chunks, vectors):

        points = []

        for i, chunk in enumerate(chunks):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vectors[i].tolist(),
                    payload={
                        "file_id": file_id,
                        "title": title,
                        "chunk_index": i,
                        "text": chunk
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

        return len(points)

    def search(self, query_vector, top_k=5):

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=top_k
        ).points

        return [
            {
                "title": r.payload["title"],
                "text": r.payload["text"],
                "score": round(r.score, 3)
            }
            for r in results
        ]
    def collection_stats(self):
        info = self.client.get_collection(self.collection)

        return {
        "collection": self.collection,
        "vectors_count": info.points_count
    }