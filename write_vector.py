"""
Writes a text chunk + its real embedding + metadata into Qdrant Cloud.
"""
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from embed import embed

VECTOR_SIZE = 768  # nomic-embed-text output dimension


def _ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def _chunk_id(text: str) -> int:
    """Stable numeric ID from the text content, so re-running on the same
    chunk updates it instead of creating a duplicate point."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def write_to_vector_store(text: str, source: str = "", date: str = ""):
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    _ensure_collection(client)

    vector = embed(text)

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            PointStruct(
                id=_chunk_id(text),
                vector=vector,
                payload={"text": text, "source": source, "date": date},
            )
        ],
    )
    print(f"  Wrote 1 chunk to Qdrant (source: {source}).")