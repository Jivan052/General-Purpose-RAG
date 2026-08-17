import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.config import settings


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    # Embedded mode: Qdrant runs in-process and persists to a local folder.
    # No server, no Docker.
    return QdrantClient(path=settings.qdrant_path)


def clear_client_cache() -> None:
    get_client.cache_clear()


def ensure_collection() -> None:
    """Stage 2: make sure the rag_documents collection exists."""
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )


def upsert_chunks(chunks: list[str], vectors: list[list[float]], source: str) -> int:
    """
    Store vector + text + source + chunk_id for each chunk.
    Returns the number of points inserted.
    """
    ensure_collection()
    client = get_client()

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": chunk,
                "source": source,
                "chunk_id": idx,
            },
        )
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)


def search(query_vector: list[float], top_k: int) -> list[dict]:
    """Stage 3: similarity search — returns top-k unique chunks by text."""
    ensure_collection()
    client = get_client()

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=max(top_k * 3, top_k),
        with_payload=True,
    ).points

    unique: list[dict] = []
    seen_texts: set[str] = set()

    for r in results:
        text = str(r.payload.get("text", "")).strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        unique.append({
            "text": text,
            "source": r.payload.get("source", ""),
            "chunk_id": r.payload.get("chunk_id"),
            "score": round(r.score, 4),
        })
        if len(unique) >= top_k:
            break

    return unique
