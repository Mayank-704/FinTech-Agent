"""
Aegis – Qdrant vector DB client for semantic user preferences.
"""
from __future__ import annotations

from typing import Optional
import structlog

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from backend.core.config import settings

log = structlog.get_logger()


def get_qdrant_client() -> Optional[QdrantClient]:
    """Return a Qdrant client, or None if unavailable (graceful degradation)."""
    try:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
            timeout=5.0,
        )
        # Verify connectivity
        client.get_collections()
        return client
    except Exception as exc:
        log.warning("qdrant_unavailable", error=str(exc))
        return None


def ensure_collection(client: QdrantClient, collection: str = settings.QDRANT_COLLECTION):
    """Create collection if it doesn't exist."""
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        log.info("qdrant_collection_created", collection=collection)


def upsert_preference(
    client: QdrantClient,
    user_id: str,
    preference_text: str,
    embedding: list[float],
):
    """Store a user semantic preference vector."""
    import hashlib, json

    point_id = int(hashlib.md5(f"{user_id}:{preference_text}".encode()).hexdigest(), 16) % (2**63)
    client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={"user_id": user_id, "text": preference_text},
            )
        ],
    )


def search_preferences(
    client: QdrantClient,
    embedding: list[float],
    user_id: str,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve top-K semantically similar preferences for a user."""
    results = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=embedding,
        query_filter={
            "must": [{"key": "user_id", "match": {"value": user_id}}]
        },
        limit=top_k,
    )
    return [{"text": r.payload["text"], "score": r.score} for r in results]
