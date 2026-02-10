"""Tests for the knowledge base search tool wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest  # noqa: F401

from it_agent.tools.knowledge import search_knowledge_base

# --- Helpers ---

def _make_settings():
    s = MagicMock()
    s.gemini_api_key = "test-key"
    s.qdrant_url = "http://localhost:6333"
    s.qdrant_collection = "it_kb"
    return s


# --- Tests ---

@pytest.mark.asyncio
async def test_search_knowledge_base_no_settings():
    result = await search_knowledge_base("VPN")
    assert result == {"error": "Settings not configured"}


@pytest.mark.asyncio
async def test_search_knowledge_base_returns_results():
    settings = _make_settings()
    mock_result = {
        "results": [
            {"id": "KB001", "title": "VPN Setup", "content": "...", "score": 0.95}
        ],
        "count": 1,
        "error": None,
    }

    with patch("it_agent.tools.knowledge.search_kb", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_result
        with patch("it_agent.tools.knowledge.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            result = await search_knowledge_base("VPN", _settings=settings)

    assert result["count"] == 1
    assert result["results"][0]["id"] == "KB001"


@pytest.mark.asyncio
async def test_search_knowledge_base_no_results():
    settings = _make_settings()
    mock_result = {"results": [], "count": 0, "error": None}

    with patch("it_agent.tools.knowledge.search_kb", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_result
        with patch("it_agent.tools.knowledge.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            result = await search_knowledge_base("obscure", _settings=settings)

    assert result["results"] == []
    assert "No matching articles" in result["message"]


@pytest.mark.asyncio
async def test_search_knowledge_base_qdrant_error():
    settings = _make_settings()
    mock_result = {"results": [], "count": 0, "error": "Connection refused"}

    with patch("it_agent.tools.knowledge.search_kb", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_result
        with patch("it_agent.tools.knowledge.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            result = await search_knowledge_base("VPN", _settings=settings)

    assert "temporarily unavailable" in result["message"]
    assert result["results"] == []
