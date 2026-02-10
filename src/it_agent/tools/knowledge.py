"""Knowledge base search tool wrapper — semantic search via Qdrant."""

from __future__ import annotations

from google import genai

from it_agent.config import Settings
from it_agent.kb.search import search_kb


async def search_knowledge_base(
    query: str,
    n_results: int = 3,
    _settings: Settings | None = None,
    **_,
) -> dict:
    """Search the IT knowledge base via Qdrant semantic search."""
    if _settings is None:
        return {"error": "Settings not configured"}

    client = genai.Client(api_key=_settings.gemini_api_key)
    result = await search_kb(query, _settings, client, limit=n_results)

    if result.get("error"):
        return {
            "results": [],
            "message": "Knowledge base search is temporarily unavailable.",
        }

    if not result["results"]:
        return {"results": [], "message": "No matching articles found in the knowledge base."}

    return {"results": result["results"], "count": result["count"]}
