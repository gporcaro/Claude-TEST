"""Tests for the Gemini embedding helper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from it_agent.kb.embeddings import _extract_vectors, embed_texts

# --- Helpers ---

def _mock_embedding(values):
    """Create a mock embedding object with a .values attribute."""
    emb = MagicMock()
    emb.values = values
    return emb


def _make_genai_client(embeddings=None, embedding=None, side_effect=None):
    """Build a mock Gemini client with aio.models.embed_content."""
    client = MagicMock()
    response = MagicMock()

    if embeddings is not None:
        response.embeddings = [_mock_embedding(v) for v in embeddings]
        response.embedding = None
    elif embedding is not None:
        response.embeddings = None
        response.embedding = _mock_embedding(embedding)
    else:
        response.embeddings = None
        response.embedding = None

    embed_mock = AsyncMock(return_value=response, side_effect=side_effect)
    client.aio.models.embed_content = embed_mock
    return client, embed_mock


# --- _extract_vectors ---

def test_extract_vectors_batch():
    response = MagicMock()
    response.embeddings = [_mock_embedding([1.0, 2.0]), _mock_embedding([3.0, 4.0])]
    response.embedding = None
    result = _extract_vectors(response)
    assert result == [[1.0, 2.0], [3.0, 4.0]]


def test_extract_vectors_single():
    response = MagicMock()
    response.embeddings = []
    response.embedding = _mock_embedding([5.0, 6.0])
    result = _extract_vectors(response)
    assert result == [[5.0, 6.0]]


def test_extract_vectors_empty():
    response = MagicMock()
    response.embeddings = []
    response.embedding = None
    result = _extract_vectors(response)
    assert result == []


# --- embed_texts ---

@pytest.mark.asyncio
async def test_embed_texts_single_batch():
    client, mock_embed = _make_genai_client(
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    result = await embed_texts(client, ["hello", "world"])
    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    mock_embed.assert_awaited_once()


@pytest.mark.asyncio
async def test_embed_texts_multiple_batches():
    """With batch_size=2 and 3 texts, should make 2 calls."""
    call_count = 0

    async def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        batch = kwargs.get("contents", [])
        resp = MagicMock()
        resp.embeddings = [_mock_embedding([float(call_count)] * 3) for _ in batch]
        resp.embedding = None
        return resp

    client = MagicMock()
    client.aio.models.embed_content = AsyncMock(side_effect=side_effect)

    result = await embed_texts(client, ["a", "b", "c"], batch_size=2)
    assert len(result) == 3
    assert call_count == 2


@pytest.mark.asyncio
async def test_embed_texts_empty():
    client, _ = _make_genai_client(embeddings=[])
    result = await embed_texts(client, [])
    assert result == []


@pytest.mark.asyncio
async def test_embed_texts_retry_on_failure():
    """Should retry and succeed on second attempt."""
    good_response = MagicMock()
    good_response.embeddings = [_mock_embedding([1.0, 2.0])]
    good_response.embedding = None

    client = MagicMock()
    client.aio.models.embed_content = AsyncMock(
        side_effect=[RuntimeError("API error"), good_response]
    )

    result = await embed_texts(client, ["test"])
    assert len(result) == 1
    assert result[0] == [1.0, 2.0]
    assert client.aio.models.embed_content.await_count == 2


@pytest.mark.asyncio
async def test_embed_texts_all_retries_exhausted():
    """Should raise after max_attempts failures."""
    client = MagicMock()
    client.aio.models.embed_content = AsyncMock(
        side_effect=RuntimeError("persistent error")
    )

    with pytest.raises(RuntimeError, match="persistent error"):
        await embed_texts(client, ["test"])
