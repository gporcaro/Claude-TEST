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


def format_public_article_blocks(articles: list[dict]) -> list[dict]:
    """Build Slack blocks for public articles with feedback buttons."""
    blocks: list[dict] = []
    for article in articles:
        blocks.append({"type": "divider"})

        confidence = article.get("confidence_score", 0)
        status = article.get("status", "pending")
        status_label = {
            "trusted": "Trusted",
            "approved": "Approved",
            "curated": "Curated",
        }.get(status, "New")
        confidence_str = f"+{confidence}" if confidence > 0 else str(confidence)

        text = (
            f":globe_with_meridians: *{article['title']}*\n"
            f"{article.get('snippet', '')}\n"
            f"Source: {article.get('source_domain', '')} | "
            f"{status_label} ({confidence_str})\n"
            f"<{article['url']}>"
        )
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": text},
        })

        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Helpful"},
                    "action_id": "article_helpful",
                    "value": str(article["id"]),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Not helpful"},
                    "action_id": "article_not_helpful",
                    "value": str(article["id"]),
                    "style": "danger",
                },
            ],
        })
    return blocks


def format_approval_blocks(article: dict) -> list[dict]:
    """Build Slack blocks for an IT approval request."""
    text = (
        f":new: *New public article needs approval*\n\n"
        f"*Title:* {article.get('title', 'Untitled')}\n"
        f"*URL:* <{article['url']}>\n"
        f"*Source:* {article.get('source_domain', 'unknown')}\n\n"
        f"This article was found via web search. Approve to index it "
        f"for future use, or deny to block it."
    )
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": text},
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve"},
                    "action_id": "approve_article",
                    "value": str(article["id"]),
                    "style": "primary",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Deny"},
                    "action_id": "deny_article",
                    "value": str(article["id"]),
                    "style": "danger",
                },
            ],
        },
    ]


def format_recommendation_approval_blocks(
    approval_id: int,
    recommendations: list[dict],
    original_text: str,
    user_name: str,
) -> list[dict]:
    """Build Slack Block Kit blocks for a recommendation approval request in #it-helpdesk."""
    rec_lines = "\n".join(
        f"• `{r['canonical_form']}` ({r.get('category', 'general')})"
        for r in recommendations
    )
    # Truncate preview to 500 chars
    preview = original_text[:500]
    if len(original_text) > 500:
        preview += "..."

    text = (
        f":mag: *Recommendation approval needed*\n\n"
        f"*User:* {user_name}\n\n"
        f"*Recommendations requiring approval:*\n{rec_lines}\n\n"
        f"*Response preview:*\n>>>{preview}"
    )
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": text},
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve"},
                    "action_id": "approve_recommendation",
                    "value": str(approval_id),
                    "style": "primary",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Deny"},
                    "action_id": "deny_recommendation",
                    "value": str(approval_id),
                    "style": "danger",
                },
            ],
        },
    ]


def redact_recommendations(text: str, recommendations: list[dict]) -> str:
    """Remove recommendation text spans from the response and append a placeholder.

    Each recommendation dict should have an ``original_text`` key with the
    exact span to remove from the response.
    """
    redacted = text
    for rec in recommendations:
        span = rec.get("original_text", "")
        if span and span in redacted:
            redacted = redacted.replace(span, "")

    # Clean up leftover blank lines
    while "\n\n\n" in redacted:
        redacted = redacted.replace("\n\n\n", "\n\n")
    redacted = redacted.rstrip()

    placeholder = (
        "\n\n:hourglass_flowing_sand: _Specific troubleshooting steps are "
        "pending IT review. You'll be notified once approved._"
    )
    return redacted + placeholder


def format_debug_blocks(
    source: str,
    user_id: str,
    user_message: str,
    steps: list[str],
    thread_link: str = "",
    incident_channel_id: str = "",
    ticket_id: str = "",
) -> list[dict]:
    """Build Block Kit blocks for a debug reasoning trace."""
    source_emoji = {"dm": ":speech_balloon:", "incident": ":rotating_light:", "help-it": ":raising_hand:"}.get(source, ":mag:")
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{source_emoji} Debug: {source} interaction", "emoji": True},
        },
    ]

    # Context line: user + links
    ctx_parts = [f"User: <@{user_id}>"]
    if thread_link:
        ctx_parts.append(f"<{thread_link}|Original thread>")
    if incident_channel_id:
        ctx_parts.append(f"Incident: <#{incident_channel_id}>")
    if ticket_id:
        ctx_parts.append(f"Ticket: {ticket_id}")
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": " | ".join(ctx_parts)}],
    })

    # User message preview (300 chars, blockquoted)
    preview = user_message[:300]
    if len(user_message) > 300:
        preview += "..."
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": f">{preview}"},
    })

    blocks.append({"type": "divider"})

    # Numbered reasoning steps (cap at 2900 chars for Slack block limit)
    steps_text = "\n".join(steps)
    if len(steps_text) > 2900:
        steps_text = steps_text[:2900] + "\n_(truncated)_"
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": steps_text},
    })

    return blocks


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
