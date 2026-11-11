"""Access to the Qdrant vector database."""
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app import config


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    # location accepts a URL ("http://localhost:6333") or ":memory:" (used in tests)
    return QdrantClient(location=config.QDRANT_URL)


def ensure_collection() -> None:
    """Create the collection if it does not exist yet."""
    client = get_qdrant_client()
    if not client.collection_exists(config.COLLECTION_NAME):
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
        )
