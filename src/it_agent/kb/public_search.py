"""Search public vendor KB: Qdrant collection + Google CSE fallback."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import httpx
from qdrant_client import QdrantClient

from it_agent.config import Settings
from it_agent.kb.embeddings import embed_texts

logger = logging.getLogger(__name__)

_MIN_SCORE = 0.70

# Vendor support domains we trust
VENDOR_DOMAINS = [
    "support.apple.com",
    "support.microsoft.com",
    "dell.com",
    "www.dell.com",
]


async def search_public_kb(
    query: str,
    settings: Settings,
    genai_client,
    limit: int = 5,
) -> dict:
    """Search the Qdrant ``it_kb_public`` collection.

    Same dedup/threshold pattern as the internal KB search.
    Returns ``{results: [...], count: int, error: str|None}``.
    """
    try:
        vectors = await embed_texts(genai_client, [query])
        if not vectors:
            return {"results": [], "count": 0, "error": "Failed to embed query"}
        query_vector = vectors[0]

        qclient = QdrantClient(url=settings.qdrant_url)

        # Check if collection exists before searching
        existing = [c.name for c in qclient.get_collections().collections]
        if settings.qdrant_public_collection not in existing:
            return {"results": [], "count": 0, "error": None}

        response = qclient.query_points(
            collection_name=settings.qdrant_public_collection,
            query=query_vector,
            limit=limit * 3,
        )
        hits = response.points

        seen: dict[int, dict] = {}
        for hit in hits:
            if hit.score < _MIN_SCORE:
                continue
            article_id = hit.payload.get("article_id")
            if article_id is None:
                continue
            if article_id not in seen or hit.score > seen[article_id]["score"]:
                seen[article_id] = {
                    "id": article_id,
                    "title": hit.payload.get("title", ""),
                    "url": hit.payload.get("url", ""),
                    "snippet": hit.payload.get("content", ""),
                    "source_domain": hit.payload.get("source_domain", ""),
                    "score": round(hit.score, 4),
                }

        results = sorted(seen.values(), key=lambda r: r["score"], reverse=True)[:limit]
        return {"results": results, "count": len(results), "error": None}

    except Exception as e:
        logger.exception("Public KB search failed")
        return {"results": [], "count": 0, "error": str(e)}


def _is_vendor_domain(url: str) -> bool:
    """Check if a URL belongs to an allowed vendor support domain."""
    try:
        host = urlparse(url).hostname or ""
        return any(host == d or host.endswith(f".{d}") for d in VENDOR_DOMAINS)
    except Exception:
        return False


async def google_cse_search(
    query: str,
    settings: Settings,
    limit: int = 5,
) -> list[dict]:
    """Search Google Custom Search Engine, filtered to vendor support domains.

    Returns a list of ``{url, title, snippet}`` dicts.
    """
    if not settings.google_cse_api_key or not settings.google_cse_cx:
        logger.debug("Google CSE not configured, skipping web search")
        return []

    site_restrict = " OR ".join(f"site:{d}" for d in VENDOR_DOMAINS)
    full_query = f"{query} ({site_restrict})"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": settings.google_cse_api_key,
                    "cx": settings.google_cse_cx,
                    "q": full_query,
                    "num": min(limit, 10),
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        logger.exception("Google CSE search failed")
        return []

    results: list[dict] = []
    for item in data.get("items", []):
        url = item.get("link", "")
        if not _is_vendor_domain(url):
            continue
        domain = urlparse(url).hostname or ""
        results.append({
            "url": url,
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "source_domain": domain,
        })

    return results[:limit]
