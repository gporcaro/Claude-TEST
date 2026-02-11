"""Semantic search over the Qdrant KB index."""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient

from it_agent.config import Settings
from it_agent.kb.embeddings import embed_texts

logger = logging.getLogger(__name__)


async def search_kb(
    query: str,
    settings: Settings,
    genai_client,
    limit: int = 5,
) -> dict:
    """Embed *query* and search Qdrant for the most relevant KB articles.

    Over-fetches 3x limit to handle multiple chunks per article, then
    deduplicates by article_id keeping the best-scoring chunk.

    Returns {results: [...], count: int, error: str|None}.
    """
    try:
        # 1. Embed query
        vectors = await embed_texts(genai_client, [query])
        if not vectors:
            return {"results": [], "count": 0, "error": "Failed to embed query"}
        query_vector = vectors[0]

        # 2. Search Qdrant (over-fetch to handle chunk dedup)
        qclient = QdrantClient(url=settings.qdrant_url)
        response = qclient.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=limit * 3,
        )
        hits = response.points

        # 3. Deduplicate by article_id, keep best score per article
        #    Discard results below the relevance threshold.
        _MIN_SCORE = 0.60
        seen: dict[str, dict] = {}
        for hit in hits:
            if hit.score < _MIN_SCORE:
                continue
            article_id = hit.payload.get("article_id", "")
            if article_id not in seen or hit.score > seen[article_id]["score"]:
                seen[article_id] = {
                    "id": article_id,
                    "title": hit.payload.get("title", ""),
                    "content": hit.payload.get("content", ""),
                    "category": hit.payload.get("category", ""),
                    "source": hit.payload.get("source", ""),
                    "score": round(hit.score, 4),
                }

        # Sort by score descending and trim to requested limit
        results = sorted(seen.values(), key=lambda r: r["score"], reverse=True)[:limit]

        return {"results": results, "count": len(results), "error": None}

    except Exception as e:
        logger.exception("Qdrant search failed")
        return {"results": [], "count": 0, "error": str(e)}
