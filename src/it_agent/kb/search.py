"""Semantic search over the Qdrant KB index."""

from __future__ import annotations

import json
import logging

from google import genai
from qdrant_client import QdrantClient

from it_agent.config import Settings
from it_agent.kb.embeddings import embed_texts

logger = logging.getLogger(__name__)

_MIN_SCORE = 0.70
_RELEVANCE_THRESHOLD = 6  # Gemini relevance score 0-10; drop results below this


async def search_kb(
    query: str,
    settings: Settings,
    genai_client,
    limit: int = 5,
) -> dict:
    """Embed *query* and search Qdrant for the most relevant KB articles.

    Over-fetches 3x limit to handle multiple chunks per article, then
    deduplicates by article_id keeping the best-scoring chunk.  A Gemini
    relevance filter removes results that are semantically close but
    topically unrelated to the query.

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
        #    Discard results below the cosine similarity threshold.
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

        # 4. Gemini relevance filter — drop results that are topically unrelated
        if results:
            results = await _filter_by_relevance(query, results, genai_client, settings)

        return {"results": results, "count": len(results), "error": None}

    except Exception as e:
        logger.exception("Qdrant search failed")
        return {"results": [], "count": 0, "error": str(e)}


async def _filter_by_relevance(
    query: str,
    results: list[dict],
    genai_client,
    settings: Settings,
) -> list[dict]:
    """Use Gemini to score each result's relevance to the query and drop low scorers."""
    try:
        articles_desc = "\n".join(
            f'{i}. "{r["title"]}" — {r["content"][:200]}'
            for i, r in enumerate(results)
        )
        client = genai.Client(api_key=settings.gemini_api_key)
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=(
                "Rate how relevant each KB article is to the user's query.\n"
                "Score each 0-10 where 10 = directly answers the query, "
                "0 = completely unrelated topic.\n"
                "An article that mentions the query topic in passing but is "
                "mainly about something else should score 3 or below.\n\n"
                f"User query: \"{query}\"\n\n"
                f"Articles:\n{articles_desc}\n\n"
                "Return ONLY a JSON array of integers (scores in the same order "
                "as the articles). Example: [8, 2, 9]"
            ),
        )
        raw = (response.text or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        scores = json.loads(raw)
        if not isinstance(scores, list) or len(scores) != len(results):
            logger.debug("Relevance filter returned unexpected shape, skipping filter")
            return results

        filtered = []
        for result, score in zip(results, scores):
            if isinstance(score, (int, float)) and score >= _RELEVANCE_THRESHOLD:
                filtered.append(result)
            else:
                logger.info(
                    "Filtered out KB result '%s' (relevance %s < %d) for query '%s'",
                    result["title"], score, _RELEVANCE_THRESHOLD, query[:80],
                )
        return filtered
    except Exception:
        logger.debug("Gemini relevance filter failed, returning unfiltered results", exc_info=True)
        return results
