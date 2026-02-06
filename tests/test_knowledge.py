"""Tests for knowledge base indexer and search."""

from __future__ import annotations

import pytest

from it_agent.knowledge.indexer import _split_markdown, index_docs
from it_agent.knowledge.store import search


class TestSplitMarkdown:
    def test_splits_on_headers(self):
        text = "# Title\nSome content\n## Section\nMore content"
        chunks = _split_markdown(text, "test")
        assert len(chunks) == 2
        assert "Title" in chunks[0]["content"]
        assert "Section" in chunks[1]["content"]

    def test_preserves_metadata(self):
        text = "# My Doc\nContent here"
        chunks = _split_markdown(text, "my-doc")
        assert chunks[0]["metadata"]["source"] == "my-doc"
        assert chunks[0]["metadata"]["title"] == "My Doc"

    def test_empty_text(self):
        chunks = _split_markdown("", "empty")
        assert chunks == []

    def test_no_headers(self):
        text = "Just some plain text without headers"
        chunks = _split_markdown(text, "plain")
        assert len(chunks) == 1


@pytest.mark.asyncio
async def test_index_and_search(tmp_path):
    # Create test docs
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "test.md").write_text("# VPN Setup\nHow to set up VPN on your laptop")

    chroma_dir = tmp_path / "chroma"

    # Index
    count = index_docs(docs_dir, chroma_dir)
    assert count > 0

    # Search
    results = await search(chroma_dir, "VPN setup", n_results=1)
    assert len(results) > 0
    assert "VPN" in results[0]["content"]


@pytest.mark.asyncio
async def test_search_empty_collection(tmp_path):
    chroma_dir = tmp_path / "empty_chroma"
    results = await search(chroma_dir, "anything", n_results=1)
    assert results == []
