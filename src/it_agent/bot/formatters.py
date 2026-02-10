"""Slack Block Kit formatting helpers."""

from __future__ import annotations

import re


def linkify_servicenow_refs(text: str, sn_base_url: str) -> str:
    """Replace INC\\d+ and KB\\d+ references with clickable Slack links.

    Preserves existing Slack links ``<url|label>`` by only processing text
    segments outside of angle-bracket pairs.
    """
    base = sn_base_url.rstrip("/")

    def _linkify_segment(segment: str) -> str:
        segment = re.sub(
            r"\b(INC\d+)\b",
            lambda m: f"<{base}/incident.do?sysparm_query=number={m.group(1)}|{m.group(1)}>",
            segment,
        )
        segment = re.sub(
            r"\b(KB\d+)\b",
            lambda m: f"<{base}/kb_view.do?sysparm_article={m.group(1)}|{m.group(1)}>",
            segment,
        )
        return segment

    # Split on existing Slack links (<...>) and only transform non-link parts
    parts = re.split(r"(<[^>]+>)", text)
    return "".join(
        part if part.startswith("<") else _linkify_segment(part)
        for part in parts
    )


def format_response_blocks(text: str, sn_base_url: str = "") -> list[dict]:
    """Format a plain text response into Slack blocks."""
    if sn_base_url:
        text = linkify_servicenow_refs(text, sn_base_url)
    blocks = []
    # Split long messages into multiple section blocks (Slack limit: 3000 chars per block)
    chunks = _chunk_text(text, 3000)
    for chunk in chunks:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": chunk}})
    return blocks


def format_error_blocks(error: str) -> list[dict]:
    """Format an error message into Slack blocks."""
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":warning: *Error:* {error}",
            },
        }
    ]


def format_ticket_blocks(ticket: dict) -> list[dict]:
    """Format a ticket into Slack blocks."""
    status_emoji = {
        "open": ":large_blue_circle:",
        "in_progress": ":hourglass:",
        "waiting": ":pause_button:",
        "resolved": ":white_check_mark:",
        "closed": ":lock:",
    }
    emoji = status_emoji.get(ticket.get("status", ""), ":ticket:")

    fields = [
        f"*ID:* {ticket['id']}",
        f"*Status:* {emoji} {ticket['status']}",
        f"*Priority:* {ticket.get('priority', 'N/A')}",
        f"*Category:* {ticket.get('category', 'N/A')}",
    ]

    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{ticket['title']}*"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(fields)},
        },
    ]


def _chunk_text(text: str, max_len: int) -> list[str]:
    """Split text into chunks respecting word boundaries."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Find last newline or space within limit
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = text.rfind(" ", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    return chunks
