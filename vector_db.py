import os

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    def __init__(self, url=None, api_key=None, collection="docs", dim=384):
        # Falls back to local Docker Qdrant for dev; in production, set
        # QDRANT_URL (e.g. your Qdrant Cloud cluster URL) and QDRANT_API_KEY.
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = api_key or os.getenv("QDRANT_API_KEY")  # None is fine for local, no auth needed

        self.client = QdrantClient(url=url, api_key=api_key, timeout=30)
        self.collection = collection

        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            existing_dim = info.config.params.vectors.size
            if existing_dim != dim:
                # The collection was built for a different embedding model
                # (e.g. OpenAI's 3072-dim embeddings). Vectors from that model
                # are incompatible with the new dim, so it must be rebuilt —
                # this wipes it, meaning you'll need to re-ingest your PDFs.
                self.client.delete_collection(self.collection)
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        else:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        )
        contexts = []
        sources = set()

        for r in response.points:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}