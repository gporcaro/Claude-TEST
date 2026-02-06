"""Knowledge base search tool wrapper."""

from __future__ import annotations

from it_agent.config import Settings
from it_agent.knowledge.store import search


async def search_knowledge_base(
    query: str,
    n_results: int = 3,
    _settings: Settings | None = None,
    **_,
) -> dict:
    """Search the IT knowledge base."""
    if _settings is None:
        return {"error": "Settings not configured"}

    try:
        results = await search(_settings.chroma_path, query, n_results)
        if not results:
            return {"results": [], "message": "No matching documents found in the knowledge base."}
        return {"results": results, "count": len(results)}
    except Exception as e:
        return {"error": f"Knowledge base search failed: {e}"}
