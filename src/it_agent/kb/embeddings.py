"""Gemini embedding helper with batching and retry."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_MODEL = "models/text-embedding-004"


async def embed_texts(
    client,
    texts: list[str],
    batch_size: int = 100,
) -> list[list[float]]:
    """Embed a list of texts using Gemini text-embedding-004.

    Processes in batches and retries with exponential backoff on failure.
    Returns a flat list of embedding vectors (one per input text).
    """
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = await _embed_batch_with_retry(client, batch)
        all_embeddings.extend(embeddings)

    return all_embeddings


async def _embed_batch_with_retry(
    client,
    batch: list[str],
    max_attempts: int = 3,
) -> list[list[float]]:
    """Embed a single batch with exponential backoff retry."""
    for attempt in range(max_attempts):
        try:
            response = await client.aio.models.embed_content(
                model=_MODEL,
                contents=batch,
            )
            return _extract_vectors(response)
        except Exception:
            if attempt == max_attempts - 1:
                raise
            wait = 2**attempt
            logger.warning("Embedding attempt %d failed, retrying in %ds", attempt + 1, wait)
            await asyncio.sleep(wait)
    return []  # unreachable, but keeps type checkers happy


def _extract_vectors(response) -> list[list[float]]:
    """Extract embedding vectors from the Gemini API response.

    Handles both single-embedding and batch-embedding response shapes.
    """
    if hasattr(response, "embeddings") and response.embeddings:
        return [e.values for e in response.embeddings]
    if hasattr(response, "embedding") and response.embedding is not None:
        return [response.embedding.values]
    return []
