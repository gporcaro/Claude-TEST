"""Tests for semantic KB search."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest  # noqa: F401

from it_agent.kb.search import search_kb

# --- Helpers ---

def _make_settings():
    s = MagicMock()
    s.qdrant_url = "http://localhost:6333"
    s.qdrant_collection = "it_kb"
    s.gemini_api_key = "test-key"
    return s


def _mock_embedding(values):
    emb = MagicMock()
    emb.values = values
    return emb


def _make_genai_client(vector=None):
    """Build a mock Gemini client that returns a single embedding."""
    vector = vector or [0.1] * 768
    client = MagicMock()
    response = MagicMock()
    response.embeddings = [_mock_embedding(vector)]
    response.embedding = None
    client.aio.models.embed_content = AsyncMock(return_value=response)
    return client


def _make_hit(article_id, title, score, content="Some content", chunk_index=0):
    """Create a mock Qdrant search hit."""
    hit = MagicMock()
    hit.payload = {
        "article_id": article_id,
        "title": title,
        "content": content,
        "category": "General",
        "source": "servicenow",
        "chunk_index": chunk_index,
        "total_chunks": 1,
    }
    hit.score = score
    return hit


# --- Tests ---

@pytest.mark.asyncio
async def test_search_kb_returns_results():
    settings = _make_settings()
    genai = _make_genai_client()

    hits = [
        _make_hit("KB001", "VPN Setup", 0.95),
        _make_hit("KB002", "Password Reset", 0.80),
    ]
    mock_qclient = MagicMock()
    mock_qclient.search.return_value = hits

    with patch("it_agent.kb.search.QdrantClient", return_value=mock_qclient):
        result = await search_kb("how to connect VPN", settings, genai, limit=5)

    assert result["error"] is None
    assert result["count"] == 2
    assert result["results"][0]["id"] == "KB001"
    assert result["results"][0]["score"] == 0.95


@pytest.mark.asyncio
async def test_search_kb_deduplicates_chunks():
    """Multiple chunks from same article should be deduped, keeping best score."""
    settings = _make_settings()
    genai = _make_genai_client()

    hits = [
        _make_hit("KB001", "VPN Setup", 0.95, chunk_index=0),
        _make_hit("KB001", "VPN Setup", 0.88, chunk_index=1),
        _make_hit("KB002", "Password", 0.80),
    ]
    mock_qclient = MagicMock()
    mock_qclient.search.return_value = hits

    with patch("it_agent.kb.search.QdrantClient", return_value=mock_qclient):
        result = await search_kb("VPN", settings, genai, limit=5)

    assert result["count"] == 2
    assert result["results"][0]["score"] == 0.95


@pytest.mark.asyncio
async def test_search_kb_respects_limit():
    settings = _make_settings()
    genai = _make_genai_client()

    hits = [_make_hit(f"KB{i:03d}", f"Article {i}", 0.9 - i * 0.1) for i in range(10)]
    mock_qclient = MagicMock()
    mock_qclient.search.return_value = hits

    with patch("it_agent.kb.search.QdrantClient", return_value=mock_qclient):
        result = await search_kb("query", settings, genai, limit=3)

    assert result["count"] == 3


@pytest.mark.asyncio
async def test_search_kb_empty_results():
    settings = _make_settings()
    genai = _make_genai_client()

    mock_qclient = MagicMock()
    mock_qclient.search.return_value = []

    with patch("it_agent.kb.search.QdrantClient", return_value=mock_qclient):
        result = await search_kb("obscure query", settings, genai, limit=5)

    assert result["count"] == 0
    assert result["results"] == []
    assert result["error"] is None


@pytest.mark.asyncio
async def test_search_kb_qdrant_failure():
    """Should return graceful error when Qdrant is down."""
    settings = _make_settings()
    genai = _make_genai_client()

    with patch(
        "it_agent.kb.search.QdrantClient",
        side_effect=Exception("Connection refused"),
    ):
        result = await search_kb("query", settings, genai, limit=5)

    assert result["count"] == 0
    assert result["results"] == []
    assert "Connection refused" in result["error"]


@pytest.mark.asyncio
async def test_search_kb_embedding_failure():
    """Should return error when embedding fails."""
    settings = _make_settings()
    genai = MagicMock()
    genai.aio.models.embed_content = AsyncMock(side_effect=RuntimeError("API down"))

    result = await search_kb("query", settings, genai, limit=5)

    assert result["count"] == 0
    assert result["error"] is not None
