"""Tests for Slack formatters."""

from __future__ import annotations

from it_agent.bot.formatters import (
    _chunk_text,
    format_error_blocks,
    format_response_blocks,
    format_ticket_blocks,
)


class TestChunkText:
    def test_short_text_no_split(self):
        assert _chunk_text("hello", 100) == ["hello"]

    def test_splits_on_space(self):
        text = "word " * 20  # 100 chars
        chunks = _chunk_text(text, 50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_empty_text(self):
        assert _chunk_text("", 100) == [""]


class TestFormatBlocks:
    def test_response_blocks(self):
        blocks = format_response_blocks("Hello world")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "section"
        assert blocks[0]["text"]["text"] == "Hello world"

    def test_error_blocks(self):
        blocks = format_error_blocks("Something broke")
        assert ":warning:" in blocks[0]["text"]["text"]

    def test_ticket_blocks(self):
        ticket = {
            "id": 1,
            "title": "Test",
            "status": "open",
            "priority": "high",
            "category": "software",
        }
        blocks = format_ticket_blocks(ticket)
        assert len(blocks) == 2
        assert "Test" in blocks[0]["text"]["text"]
