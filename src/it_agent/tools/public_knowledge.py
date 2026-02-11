"""Public article search tool — Qdrant first, Google CSE fallback."""

from __future__ import annotations

import logging

from google import genai

from it_agent import db
from it_agent.config import Settings
from it_agent.kb.public_search import google_cse_search, search_public_kb

logger = logging.getLogger(__name__)


async def search_public_articles(
    query: str,
    n_results: int = 3,
    _settings: Settings | None = None,
    _user_id: str = "unknown",
    **_,
) -> dict:
    """Search public vendor articles. Qdrant first, Google CSE fallback.

    Returns ``{results: [...], needs_approval: [...]}`` where *needs_approval*
    lists articles from CSE that require IT approval before being shared.
    """
    if _settings is None:
        return {"error": "Settings not configured"}

    n_results = min(max(n_results, 1), 5)
    client = genai.Client(api_key=_settings.gemini_api_key)

    # 1. Search Qdrant public collection
    qdrant_result = await search_public_kb(query, _settings, client, limit=n_results)
    results: list[dict] = []
    needs_approval: list[dict] = []

    if qdrant_result.get("results"):
        for r in qdrant_result["results"]:
            # Enrich with DB data (status, confidence, votes)
            article = await db.get_public_article(r["id"])
            if article:
                results.append({
                    "id": article["id"],
                    "title": article["title"],
                    "url": article["url"],
                    "snippet": article.get("snippet", r.get("snippet", "")),
                    "source_domain": article["source_domain"],
                    "status": article["status"],
                    "confidence_score": article["confidence_score"],
                    "score": r.get("score", 0),
                })

    # 2. If not enough results, try Google CSE
    if len(results) < n_results:
        cse_results = await google_cse_search(query, _settings, limit=n_results - len(results))

        for cse in cse_results:
            url = cse["url"]
            # Check if URL already in DB
            existing = await db.get_public_article_by_url(url)

            if existing:
                if existing["status"] == "denied":
                    continue
                if existing["status"] in ("curated", "approved", "trusted"):
                    results.append({
                        "id": existing["id"],
                        "title": existing["title"],
                        "url": existing["url"],
                        "snippet": existing.get("snippet", cse.get("snippet", "")),
                        "source_domain": existing["source_domain"],
                        "status": existing["status"],
                        "confidence_score": existing["confidence_score"],
                        "score": 0,
                    })
                elif existing["status"] == "pending":
                    needs_approval.append({
                        "article_id": existing["id"],
                        "url": existing["url"],
                        "title": existing["title"],
                    })
            else:
                # New article from web search — create as pending
                try:
                    article_id = await db.create_public_article(
                        url=url,
                        title=cse["title"],
                        snippet=cse.get("snippet", ""),
                        source_domain=cse.get("source_domain", ""),
                        status="pending",
                    )
                    needs_approval.append({
                        "article_id": article_id,
                        "url": url,
                        "title": cse["title"],
                    })
                except Exception:
                    logger.debug("Failed to create pending article for %s", url, exc_info=True)

    if not results and not needs_approval:
        return {
            "results": [],
            "needs_approval": [],
            "message": "No relevant public articles found.",
        }

    return {
        "results": results,
        "needs_approval": needs_approval,
        "count": len(results),
    }
