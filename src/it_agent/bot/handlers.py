from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone

from google import genai
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from it_agent.agent import executor
from it_agent.agent.core import Agent, AgentResult
from it_agent.bot.events import emit
from it_agent.bot.formatters import (
    format_approval_blocks,
    format_error_blocks,
    format_public_article_blocks,
    format_recommendation_approval_blocks,
    format_response_blocks,
    linkify_servicenow_refs,
    redact_recommendations,
)
from it_agent.config import Settings
from it_agent import db
from it_agent.servicenow.client import ServiceNowClient
from it_agent.kb.indexer import _strip_html
from it_agent.tools.tickets import create_incident_channel, create_ticket as _create_ticket_tool, get_ticket
from it_agent.tools.users import resolve_sn_user_to_slack

logger = logging.getLogger(__name__)

# Per-thread conversation history: {(channel, thread_ts): [messages]}
_conversations: dict[tuple[str, str], list[dict]] = {}
MAX_HISTORY = 20

# Shared agent instance
_agent: Agent | None = None

# Tracks ticket → original #help-it thread so we can post resolution updates.
# ticket_id → (channel, thread_ts)
_ticket_threads: dict[str, tuple[str, str]] = {}

# Incident channel IDs the bot should actively respond in.
_incident_channels: set[str] = set()

# Per-incident-channel context: channel_id → {ticket_id, title, description, ...}
_incident_context: dict[str, dict] = {}

# Pending channel creation: ticket_id → {user_id, ticket, thread_ts, channel}
_pending_channels: dict[str, dict] = {}

# Resolved tickets pending auto-close: ticket_id → resolved_epoch
_resolved_pending_close: dict[str, float] = {}

# Debounced live-summary timers: ticket_id → pending asyncio.Task
_summary_timers: dict[str, asyncio.Task] = {}

# Original descriptions saved at ticket creation: ticket_id → str
_ticket_original_descriptions: dict[str, str] = {}

_SUMMARY_DEBOUNCE_SECONDS = 5 * 60  # 5 minutes

# Interaction tracking: (channel, thread_ts) → interaction_id in the DB
_interaction_ids: dict[tuple[str, str], int] = {}

# 48 hours in seconds
_AUTO_CLOSE_DELAY = 48 * 60 * 60

# How often to check (1 hour)
_AUTO_CLOSE_CHECK_INTERVAL = 60 * 60

# [AI Context] KB articles loaded at startup: article_id → {id, title, content}
_ai_context_articles: dict[str, dict] = {}


async def load_ai_context_articles(settings: Settings) -> None:
    """Fetch KB articles prefixed with [AI Context] and cache them for agent use."""
    client = ServiceNowClient(
        settings.sn_instance_url, settings.sn_username, settings.sn_password,
    )
    try:
        articles = await client.list_kb_articles()
    finally:
        await client.close()

    for article in articles:
        title = article.get("title", "")
        if title.startswith("[AI Context]"):
            cleaned = {
                "id": article["id"],
                "title": title,
                "content": _strip_html(article.get("content", "")),
            }
            _ai_context_articles[article["id"]] = cleaned

    logger.info("Loaded %d AI context article(s)", len(_ai_context_articles))


def _get_agent(settings: Settings) -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent(settings)
    return _agent


async def _resolve_user_name(user_id: str, settings: Settings) -> str:
    """Best-effort resolve Slack user_id to display name."""
    try:
        client = AsyncWebClient(token=settings.slack_bot_token)
        info = await client.users_info(user=user_id)
        profile = info["user"]["profile"]
        return profile.get("display_name") or profile.get("real_name") or user_id
    except Exception:
        return user_id


async def _record_agent_result(
    interaction_id: int, user_text: str, result: AgentResult,
) -> None:
    """Record messages and tool calls from an agent run into the database."""
    try:
        await db.add_message(interaction_id, "user", user_text)
        for tc in result.tool_calls:
            summary = ""
            r = tc.get("result")
            if isinstance(r, dict):
                if r.get("success") is not None:
                    summary = "success" if r["success"] else "failed"
                elif r.get("error"):
                    summary = f"error: {str(r['error'])[:100]}"
            elif isinstance(r, str):
                summary = r[:200]
            await db.add_tool_call(
                interaction_id, tc["name"], tc.get("args"), summary,
            )
        await db.add_message(interaction_id, "assistant", result.text)
        await db.increment_counts(
            interaction_id,
            messages=2,  # user + assistant
            tool_calls=len(result.tool_calls),
        )
    except Exception:
        logger.debug("Failed to record agent result for interaction %d", interaction_id, exc_info=True)


async def _recover_thread_history(
    channel: str, thread_ts: str, settings: Settings, bot_user_id: str | None = None,
) -> list[dict]:
    """Fetch a Slack thread's messages and rebuild conversation history.

    Called when the bot has no in-memory history for a thread (e.g. after restart).
    Returns a list of ``{role, content}`` dicts ready for the agent.
    """
    try:
        client = AsyncWebClient(token=settings.slack_bot_token)

        # Resolve bot's own user ID so we can tag its messages as "assistant"
        if bot_user_id is None:
            auth = await client.auth_test()
            bot_user_id = auth["user_id"]

        resp = await client.conversations_replies(
            channel=channel, ts=thread_ts, limit=MAX_HISTORY,
        )
        messages = resp.get("messages", [])

        history: list[dict] = []
        for msg in messages:
            # Skip subtypes (joins, topic changes, etc.)
            if msg.get("subtype"):
                continue
            text = msg.get("text", "").strip()
            if not text:
                continue

            if msg.get("user") == bot_user_id or msg.get("bot_id"):
                history.append({"role": "assistant", "content": text})
            else:
                history.append({"role": "user", "content": text})

        if history:
            logger.info(
                "Recovered %d messages for thread %s in channel %s",
                len(history), thread_ts, channel,
            )
        return history
    except Exception:
        logger.warning("Failed to recover thread history", exc_info=True)
        return []


def _strip_channel_created_lines(text: str, channel_name: str) -> str:
    """Remove lines about the private channel being created from agent text.

    Keeps ticket/incident references but strips the "a private channel #xyz
    has been created" boilerplate so the message makes sense when posted
    *inside* that channel.
    """
    filtered: list[str] = []
    for line in text.split("\n"):
        lower = line.lower()
        # Skip lines that talk about channel creation
        if "private channel" in lower and "created" in lower:
            continue
        if channel_name and f"#{channel_name}" in lower and "created" in lower:
            continue
        if "has been created to troubleshoot" in lower:
            continue
        filtered.append(line)
    # Collapse multiple blank lines that may result from stripping
    result = "\n".join(filtered)
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    return result.strip()


def _parse_incident_message(text: str) -> dict:
    """Extract incident fields from the bot's initial channel message."""
    ctx: dict[str, str] = {}
    m = re.search(r"\*Incident (INC\d+)\*", text)
    if m:
        ctx["ticket_id"] = m.group(1)
    m = re.search(r"\*Title:\*\s*(.+)", text)
    if m:
        ctx["title"] = m.group(1).strip()
    m = re.search(r"\*Priority:\*\s*(\w+)", text)
    if m:
        ctx["priority"] = m.group(1).strip()
    m = re.search(r"\*Description:\*\s*(.+?)(?:\n|$)", text)
    if m:
        ctx["description"] = m.group(1).strip()
    return ctx


def _ticket_id_from_channel_name(channel_name: str) -> str | None:
    """Extract a ticket ID from an incident channel name.

    Channel names follow the pattern ``inc0129540-username``.
    Returns e.g. ``'INC0129540'`` or *None* if the name doesn't match.
    """
    m = re.match(r"^(inc\d+)", channel_name, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


_ESCALATION_PATTERNS = re.compile(
    r"\b(human|person|someone|escalat|talk\s+to|speak\s+to|real\s+person|"
    r"transfer|assign|technician|agent|help\s+desk|helpdesk)\b",
    re.IGNORECASE,
)


async def _unassign_bot_on_escalation(ticket_id: str, settings: Settings) -> None:
    """Remove bot as primary assignee, keep as additional assignee."""
    if not settings.sn_bot_user_sys_id:
        return
    client = ServiceNowClient(
        settings.sn_instance_url, settings.sn_username, settings.sn_password,
    )
    try:
        incident = await client.get_incident(ticket_id)
        if incident is None:
            return
        # Only unassign if bot is currently the primary assignee
        if incident.get("assignee_id") != settings.sn_bot_user_sys_id:
            return
        await client.update_incident(
            incident["sys_id"],
            {
                "assignee_id": "",
                "additional_assignee_list": settings.sn_bot_user_sys_id,
            },
            current_state=incident.get("_raw_state", "1"),
        )
        logger.info("Unassigned bot from %s, added as additional assignee", ticket_id)
    finally:
        await client.close()


def _recover_ticket_thread_mapping(
    channel: str, thread_ts: str, history: list[dict],
) -> str | None:
    """Scan recovered thread messages for a ticket ID and repopulate _ticket_threads.

    Returns the ticket_id if found, or None.
    """
    for msg in history:
        # Look for INC numbers in bot messages (from the follow-up or context)
        m = re.search(r"(INC\d{7,})", msg["content"])
        if m:
            ticket_id = m.group(1)
            if ticket_id not in _ticket_threads:
                _ticket_threads[ticket_id] = (channel, thread_ts)
                logger.info(
                    "Recovered ticket-thread mapping: %s → (%s, %s)",
                    ticket_id, channel, thread_ts,
                )
            return ticket_id
    return None


def _find_incident_channel_for_thread(channel: str, thread_ts: str) -> str | None:
    """Reverse-lookup: find the incident channel associated with a #help-it thread.

    Walks _ticket_threads to find the ticket_id for (channel, thread_ts),
    then walks _incident_context to find the channel_id for that ticket.
    """
    ticket_id: str | None = None
    for tid, (ch, ts) in _ticket_threads.items():
        if ch == channel and ts == thread_ts:
            ticket_id = tid
            break
    if ticket_id is None:
        return None
    for ch_id, ctx in _incident_context.items():
        if ctx.get("ticket_id") == ticket_id:
            return ch_id
    return None


async def discover_incident_channels(settings: Settings) -> None:
    """Scan private channels the bot belongs to and register incident channels.

    Also fetches the first bot message in each channel to seed incident context.
    """
    try:
        client = AsyncWebClient(token=settings.slack_bot_token)
        cursor = None
        channel_ids: list[str] = []
        while True:
            resp = await client.conversations_list(
                types="private_channel", exclude_archived=True, limit=200, cursor=cursor
            )
            for ch in resp.get("channels", []):
                name = ch.get("name", "")
                if name.startswith("inc") and ch.get("is_member"):
                    _incident_channels.add(ch["id"])
                    channel_ids.append(ch["id"])
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        # Fetch context from the first bot message in each channel.
        # Fall back to extracting the ticket ID from the channel name.
        # Also track channels whose name ends with "-resolved" so we can
        # recover auto-close tracking after a restart.
        resolved_ticket_ids: list[str] = []

        for ch_id in channel_ids:
            try:
                info = await client.conversations_info(channel=ch_id)
                ch_name = info.get("channel", {}).get("name", "")

                hist = await client.conversations_history(channel=ch_id, limit=50)
                # Scan all messages for the incident summary (posted by the bot)
                found = False
                for msg in hist.get("messages", []):
                    if not msg.get("bot_id"):
                        continue
                    text = msg.get("text", "")
                    ctx = _parse_incident_message(text)
                    if ctx.get("ticket_id"):
                        # Strip any existing LIVE SUMMARY to get clean original text
                        original = text.split("\n\n*LIVE SUMMARY:*")[0]
                        _incident_context[ch_id] = {
                            **ctx,
                            "summary_ts": msg["ts"],
                            "original_text": original,
                            "summary_lines": [],
                        }
                        found = True
                        break

                # Fallback: extract ticket ID from channel name
                if not found:
                    ticket_id = _ticket_id_from_channel_name(ch_name)
                    if ticket_id:
                        _incident_context[ch_id] = {
                            "ticket_id": ticket_id,
                            "title": "",
                            "description": "",
                            "priority": "",
                            "summary_ts": None,
                            "original_text": None,
                            "summary_lines": [],
                        }
                        logger.info(
                            "Seeded incident context for %s from channel name (%s)",
                            ch_name, ticket_id,
                        )

                # Track resolved channels for auto-close recovery
                if ch_name.endswith("-resolved"):
                    tid = _incident_context.get(ch_id, {}).get("ticket_id")
                    if tid:
                        resolved_ticket_ids.append(tid)

            except Exception:
                logger.debug("Could not fetch history for channel %s", ch_id)

        # Recover auto-close tracking for resolved channels so the 48h
        # countdown survives bot restarts.
        if resolved_ticket_ids:
            sn_client = ServiceNowClient(
                settings.sn_instance_url, settings.sn_username, settings.sn_password,
            )
            try:
                for tid in resolved_ticket_ids:
                    if tid in _resolved_pending_close:
                        continue
                    try:
                        incident = await sn_client.get_incident(tid)
                        if incident is None or incident.get("status") != "resolved":
                            continue
                        updated_str = incident.get("updated_at", "")
                        if updated_str:
                            resolved_dt = datetime.strptime(
                                updated_str, "%Y-%m-%d %H:%M:%S",
                            ).replace(tzinfo=timezone.utc)
                            _resolved_pending_close[tid] = resolved_dt.timestamp()
                            logger.info(
                                "Recovered auto-close tracking for %s (resolved at %s)",
                                tid, updated_str,
                            )
                    except Exception:
                        logger.debug("Could not check SN status for %s", tid)
            finally:
                await sn_client.close()

        if _incident_channels:
            logger.info(
                "Discovered %d incident channel(s) on startup (%d with context, %d pending auto-close)",
                len(_incident_channels),
                len(_incident_context),
                len(_resolved_pending_close),
            )
    except Exception:
        logger.warning("Failed to discover incident channels", exc_info=True)


def register_handlers(app: AsyncApp, settings: Settings) -> None:
    """Register Slack event handlers."""

    @app.event("app_mention")
    async def handle_mention(event: dict, say) -> None:
        """Handle @bot mentions in channels."""
        # If the mention is in #help-it or an incident channel, skip —
        # handle_message covers those so we don't double-respond.
        channel = event.get("channel", "")
        if settings.help_channel_id and channel == settings.help_channel_id:
            return
        if channel in _incident_channels:
            return

        # Strip the bot mention from the text
        text = re.sub(r"<@[A-Z0-9]+>\s*", "", event.get("text", "")).strip()
        if not text:
            await say("Hi! I'm the IT Support Agent. How can I help you?")
            return
        await _handle_message(event, text, say, settings)

    @app.event("message")
    async def handle_message(event: dict, say) -> None:
        """Route messages: DMs → agent, #help-it → proactive handler, others → ignore."""
        # Ignore bot messages, edits, etc.
        if event.get("subtype") or event.get("bot_id"):
            return

        text = event.get("text", "").strip()
        if not text:
            return

        channel_type = event.get("channel_type", "")
        channel = event.get("channel", "")

        # DM → process as before
        if channel_type == "im":
            await _handle_message(event, text, say, settings)
            return

        # #help-it channel → proactive handler
        if settings.help_channel_id and channel == settings.help_channel_id:
            # Strip any @bot mention so we don't echo it back
            text = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()
            if not text:
                return
            await _handle_help_channel_message(event, text, say, settings)
            return

        # Incident channels → continue conversation (no threading)
        if channel in _incident_channels:
            await _handle_incident_message(event, text, say, settings)
            return

        # Other channels → ignore (handled by app_mention only)

    # Wire the resolution callback
    async def _resolution_callback(
        ticket_id: str, ticket_data: dict, settings: Settings
    ) -> None:
        await post_resolution_update(ticket_id, ticket_data, settings)

    executor._on_ticket_resolved = _resolution_callback

    async def _assignment_callback(
        ticket_id: str, assignee_id: str, settings: Settings
    ) -> None:
        await _handle_ticket_assigned(ticket_id, assignee_id, settings)

    executor._on_ticket_assigned = _assignment_callback

    # --- Public article feedback actions ---

    @app.action("article_helpful")
    async def handle_article_helpful(ack, body) -> None:
        await ack()
        article_id = body["actions"][0]["value"]
        user_id = body["user"]["id"]
        updated = await db.record_feedback(int(article_id), user_id, "helpful")
        if updated:
            # Check if article reached trust threshold
            if (
                updated["status"] not in ("trusted",)
                and updated["confidence_score"] >= settings.public_trust_threshold
            ):
                await db.update_public_article(int(article_id), status="trusted")
                logger.info("Article %s promoted to trusted (score %d)", article_id, updated["confidence_score"])
            # Update the message to acknowledge the vote
            try:
                client = AsyncWebClient(token=settings.slack_bot_token)
                pos = updated["positive_votes"]
                neg = updated["negative_votes"]
                await client.chat_update(
                    channel=body["channel"]["id"],
                    ts=body["message"]["ts"],
                    text=f"Thanks for your feedback! (Helpful: {pos} | Not helpful: {neg})",
                    blocks=body["message"].get("blocks", []),
                )
            except Exception:
                logger.debug("Failed to update feedback message", exc_info=True)

    @app.action("article_not_helpful")
    async def handle_article_not_helpful(ack, body) -> None:
        await ack()
        article_id = body["actions"][0]["value"]
        user_id = body["user"]["id"]
        updated = await db.record_feedback(int(article_id), user_id, "not_helpful")
        if updated:
            try:
                client = AsyncWebClient(token=settings.slack_bot_token)
                pos = updated["positive_votes"]
                neg = updated["negative_votes"]
                await client.chat_update(
                    channel=body["channel"]["id"],
                    ts=body["message"]["ts"],
                    text=f"Thanks for your feedback! (Helpful: {pos} | Not helpful: {neg})",
                    blocks=body["message"].get("blocks", []),
                )
            except Exception:
                logger.debug("Failed to update feedback message", exc_info=True)

    # --- Article approval actions ---

    @app.action("approve_article")
    async def handle_approve_article(ack, body) -> None:
        await ack()
        article_id = int(body["actions"][0]["value"])
        approver_id = body["user"]["id"]
        approver_name = await _resolve_user_name(approver_id, settings)

        await db.update_public_article(article_id, status="approved")

        # Index article into Qdrant
        try:
            from google import genai
            from it_agent.kb.public_indexer import index_single_article
            genai_client = genai.Client(api_key=settings.gemini_api_key)
            await index_single_article(article_id, settings, genai_client)
        except Exception:
            logger.warning("Failed to index approved article %d", article_id, exc_info=True)

        # Update the approval message
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            article = await db.get_public_article(article_id)
            title = article["title"] if article else f"Article {article_id}"
            await client.chat_update(
                channel=body["channel"]["id"],
                ts=body["message"]["ts"],
                text=f":white_check_mark: *{title}* — Approved by {approver_name}",
                blocks=[{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":white_check_mark: *{title}* — Approved by {approver_name}",
                    },
                }],
            )
        except Exception:
            logger.debug("Failed to update approval message", exc_info=True)

        await emit("article_approved", {"article_id": article_id, "approver": approver_name})

    @app.action("deny_article")
    async def handle_deny_article(ack, body) -> None:
        await ack()
        article_id = int(body["actions"][0]["value"])
        denier_id = body["user"]["id"]
        denier_name = await _resolve_user_name(denier_id, settings)

        await db.update_public_article(article_id, status="denied")

        # Update the approval message
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            article = await db.get_public_article(article_id)
            title = article["title"] if article else f"Article {article_id}"
            await client.chat_update(
                channel=body["channel"]["id"],
                ts=body["message"]["ts"],
                text=f":no_entry: *{title}* — Denied by {denier_name}",
                blocks=[{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":no_entry: *{title}* — Denied by {denier_name}",
                    },
                }],
            )
        except Exception:
            logger.debug("Failed to update denial message", exc_info=True)

        await emit("article_denied", {"article_id": article_id, "denier": denier_name})

    # --- Recommendation approval actions ---

    @app.action("approve_recommendation")
    async def handle_approve_recommendation(ack, body) -> None:
        await ack()
        approval_id = int(body["actions"][0]["value"])
        approver_id = body["user"]["id"]
        approver_name = await _resolve_user_name(approver_id, settings)

        approval = await db.get_pending_rec_approval(approval_id)
        if not approval or approval["status"] != "pending":
            return

        await db.update_pending_rec_approval(approval_id, status="approved")

        # Increment approval counts and promote to trusted if threshold reached
        rec_ids = json.loads(approval.get("recommendation_ids", "[]"))
        for rec_id in rec_ids:
            updated_rec = await db.increment_recommendation_approval(rec_id)
            if (
                updated_rec
                and updated_rec["status"] != "trusted"
                and updated_rec["approval_count"] >= settings.recommendation_trust_threshold
            ):
                await db.update_recommendation(rec_id, status="trusted")
                logger.info(
                    "Recommendation %d promoted to trusted (count %d)",
                    rec_id, updated_rec["approval_count"],
                )

        # Update the user's message with the full original response
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            sn_url = settings.sn_instance_url
            original = approval["original_text"]
            linked = linkify_servicenow_refs(original, sn_url)
            blocks = format_response_blocks(original, sn_url)
            await client.chat_update(
                channel=approval["channel_id"],
                ts=approval["message_ts"],
                text=linked,
                blocks=blocks,
            )
        except Exception:
            logger.debug("Failed to update user message on recommendation approval", exc_info=True)

        # Update the #it-helpdesk approval message
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            ch = approval.get("approval_channel") or body["channel"]["id"]
            ts = approval.get("approval_message_ts") or body["message"]["ts"]
            await client.chat_update(
                channel=ch,
                ts=ts,
                text=f":white_check_mark: Recommendations approved by {approver_name}",
                blocks=[{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":white_check_mark: Recommendations approved by {approver_name}",
                    },
                }],
            )
        except Exception:
            logger.debug("Failed to update recommendation approval message", exc_info=True)

        await emit("recommendation_approved", {
            "approval_id": approval_id, "approver": approver_name,
        })

    @app.action("deny_recommendation")
    async def handle_deny_recommendation(ack, body) -> None:
        await ack()
        approval_id = int(body["actions"][0]["value"])
        denier_id = body["user"]["id"]
        denier_name = await _resolve_user_name(denier_id, settings)

        approval = await db.get_pending_rec_approval(approval_id)
        if not approval or approval["status"] != "pending":
            return

        await db.update_pending_rec_approval(approval_id, status="denied")

        # Increment denial counts
        rec_ids = json.loads(approval.get("recommendation_ids", "[]"))
        for rec_id in rec_ids:
            await db.increment_recommendation_denial(rec_id)

        # Update the user's message with denial notice
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            denial_text = (
                f"{approval['redacted_text'].rsplit(':hourglass_flowing_sand:', 1)[0].rstrip()}"
                "\n\n:no_entry: _The specific troubleshooting steps were reviewed and not approved by IT. "
                "Please contact the help desk for further assistance._"
            )
            blocks = format_response_blocks(denial_text, settings.sn_instance_url)
            await client.chat_update(
                channel=approval["channel_id"],
                ts=approval["message_ts"],
                text=denial_text,
                blocks=blocks,
            )
        except Exception:
            logger.debug("Failed to update user message on recommendation denial", exc_info=True)

        # Update the #it-helpdesk message
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            ch = approval.get("approval_channel") or body["channel"]["id"]
            ts = approval.get("approval_message_ts") or body["message"]["ts"]
            await client.chat_update(
                channel=ch,
                ts=ts,
                text=f":no_entry: Recommendations denied by {denier_name}",
                blocks=[{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":no_entry: Recommendations denied by {denier_name}",
                    },
                }],
            )
        except Exception:
            logger.debug("Failed to update recommendation denial message", exc_info=True)

        await emit("recommendation_denied", {
            "approval_id": approval_id, "denier": denier_name,
        })

    # --- Move to private channel action ---

    @app.action("move_to_private_channel")
    async def handle_move_to_private_channel(ack, body) -> None:
        await ack()
        ticket_id = body["actions"][0]["value"]
        channel = body["channel"]["id"]
        message_ts = body["message"]["ts"]
        thread_ts = body["message"].get("thread_ts") or message_ts
        button_user_id = body["user"]["id"]

        client = AsyncWebClient(token=settings.slack_bot_token)

        # Check if channel already exists (idempotent)
        for ch_id, ctx in _incident_context.items():
            if ctx.get("ticket_id") == ticket_id:
                # Channel already created — update the button message
                try:
                    await client.chat_update(
                        channel=channel,
                        ts=message_ts,
                        text=f":white_check_mark: Private channel <#{ch_id}> has been created for {ticket_id}.",
                        blocks=[{
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f":white_check_mark: Private channel <#{ch_id}> already exists for {ticket_id}.",
                            },
                        }],
                    )
                except Exception:
                    logger.debug("Failed to update button message", exc_info=True)
                return

        # Create the channel
        result = await _create_deferred_channel(
            ticket_id, settings,
            thread_info=(channel, thread_ts),
            user_id=button_user_id,
            reason="button",
        )
        if result is None:
            try:
                await client.chat_postMessage(
                    channel=channel,
                    text=f":warning: Failed to create private channel for {ticket_id}.",
                    thread_ts=thread_ts,
                )
            except Exception:
                logger.debug("Failed to post channel creation failure message", exc_info=True)
            return

        channel_id = result["channel_id"]

        # Replace the button message with confirmation
        try:
            await client.chat_update(
                channel=channel,
                ts=message_ts,
                text=f":white_check_mark: Private channel <#{channel_id}> has been created for {ticket_id}.",
                blocks=[{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":white_check_mark: Private channel <#{channel_id}> has been created for {ticket_id}.",
                    },
                }],
            )
        except Exception:
            logger.debug("Failed to update button message", exc_info=True)

        # Post in the thread
        try:
            await client.chat_postMessage(
                channel=channel,
                text=f":arrow_right: A private channel <#{channel_id}> has been created. Further troubleshooting will continue there.",
                thread_ts=thread_ts,
            )
        except Exception:
            logger.debug("Failed to post channel created thread message", exc_info=True)

    # --- Archive channel now action ---

    @app.action("archive_channel_now")
    async def handle_archive_channel_now(ack, body) -> None:
        await ack()
        ticket_id = body["actions"][0]["value"]
        channel = body["channel"]["id"]
        message_ts = body["message"]["ts"]
        button_user_id = body["user"]["id"]

        client = AsyncWebClient(token=settings.slack_bot_token)

        # Update button message to show processing
        try:
            await client.chat_update(
                channel=channel,
                ts=message_ts,
                text=f":hourglass_flowing_sand: Archiving channel for {ticket_id}...",
                blocks=[{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":hourglass_flowing_sand: Archiving channel for {ticket_id}...",
                    },
                }],
            )
        except Exception:
            logger.debug("Failed to update archive button message", exc_info=True)

        # 1. Close ticket in ServiceNow
        sn_client = ServiceNowClient(
            settings.sn_instance_url, settings.sn_username, settings.sn_password
        )
        try:
            incident = await sn_client.get_incident(ticket_id)
            if incident and incident.get("status") in ("resolved",):
                # Resolve Slack user display name for close notes
                try:
                    user_info = await client.users_info(user=button_user_id)
                    user_name = user_info["user"].get("real_name") or user_info["user"].get("name", button_user_id)
                except Exception:
                    user_name = button_user_id
                await sn_client.update_incident(
                    incident["sys_id"],
                    {
                        "status": "closed",
                        "close_notes": f"Manually archived by {user_name} via Slack.",
                    },
                    current_state=incident.get("_raw_state", "6"),
                )
                logger.info("Closed ticket %s in ServiceNow (manual archive by %s)", ticket_id, button_user_id)
        except Exception:
            logger.warning("Archive now: failed to close %s in ServiceNow", ticket_id, exc_info=True)
        finally:
            await sn_client.close()

        # 2. Post farewell note
        sn_url = settings.sn_instance_url
        note = (
            f":lock: *This channel is now being archived.*\n\n"
            f"Ticket {ticket_id} was manually closed by <@{button_user_id}>. "
            f"If you need further assistance on this issue, "
            f"please open a new request in #help-it."
        )
        note = linkify_servicenow_refs(note, sn_url)
        try:
            await client.chat_postMessage(channel=channel, text=note)
        except Exception:
            logger.warning("Archive now: failed to post farewell note in %s", channel, exc_info=True)

        # 3. Archive the channel
        try:
            await client.conversations_archive(channel=channel)
            logger.info("Archived incident channel %s for ticket %s (manual)", channel, ticket_id)
        except Exception:
            logger.warning("Archive now: failed to archive channel %s", channel, exc_info=True)

        # 4. Update interaction record
        try:
            interaction = await db.get_interaction_by_ticket(ticket_id)
            if interaction:
                await db.update_interaction(interaction["id"], status="closed")
        except Exception:
            logger.debug("Failed to update interaction on manual archive for %s", ticket_id, exc_info=True)

        # 5. Clean up tracking dicts
        _resolved_pending_close.pop(ticket_id, None)
        _incident_channels.discard(channel)
        _incident_context.pop(channel, None)

        # 6. Emit events
        await emit("ticket_auto_closed", {"ticket_id": ticket_id, "channel_id": channel})
        await emit("channel_archived", {"ticket_id": ticket_id, "channel_id": channel})


async def _register_incident_channels(result: AgentResult) -> None:
    """Track any incident channels created during this agent run."""
    for tc in result.tool_calls:
        if tc["name"] != "create_ticket":
            continue
        r = tc.get("result", {})
        channel_id = r.get("channel_id")
        if channel_id:
            _incident_channels.add(channel_id)
            ticket = r.get("ticket", {})
            ticket_id = ticket.get("ticket_id", "")
            title = ticket.get("title", "")
            priority = ticket.get("priority", "")
            _incident_context[channel_id] = {
                "ticket_id": ticket_id,
                "title": title,
                "description": ticket.get("description", ""),
                "priority": priority,
                "summary_ts": r.get("summary_ts"),
                "original_text": None,  # will be built on first summary update
                "summary_lines": [],
            }
            # Save original description for live summary
            if ticket_id:
                _ticket_original_descriptions[ticket_id] = ticket.get(
                    "_original_description", ticket.get("description", "")
                )
            logger.info("Registered incident channel %s", channel_id)
            await emit("ticket_created", {
                "ticket_id": ticket_id,
                "title": title,
                "priority": priority,
                "channel_id": channel_id,
            })
            await emit("channel_created", {
                "channel_id": channel_id,
                "channel_name": r.get("channel_name", ""),
                "ticket_id": ticket_id,
            })


# ---------------------------------------------------------------------------
# Recommendation approval gate
# ---------------------------------------------------------------------------

_BASIC_RECOMMENDATIONS = {
    "restart", "reboot", "check cables", "clear cache", "clear browser cache",
    "try again", "log out and log back in", "sign out and sign back in",
    "check internet connection", "check your internet", "update your browser",
    "close and reopen", "power cycle", "unplug and replug",
}


async def _extract_recommendations(
    response_text: str, settings: Settings,
) -> list[dict]:
    """Use Gemini to extract actionable recommendations from the agent's response.

    Returns a list of ``{original_text, canonical_form, category}`` dicts.
    Basic suggestions (restart, check cables, clear cache) are excluded.
    Returns ``[]`` on failure so the response posts normally.
    """
    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=(
                "Analyze the following IT support response and extract ONLY specific, "
                "actionable recommendations that involve changing settings, installing "
                "software, running commands, modifying configuration, or disabling features. "
                "Do NOT include basic/generic advice like: restart, reboot, check cables, "
                "clear cache, try again, log out/in, check internet, update browser, "
                "close and reopen, power cycle.\n\n"
                "For each recommendation found, output a JSON array of objects with:\n"
                '- "original_text": the exact text span from the response\n'
                '- "canonical_form": a short normalized version (e.g. "disable hardware acceleration in Chrome")\n'
                '- "category": one of "settings_change", "install", "command", "config_change", "feature_toggle"\n\n'
                "If there are NO specific actionable recommendations, return an empty array: []\n\n"
                "IMPORTANT: Return ONLY the JSON array, no other text.\n\n"
                f"Response to analyze:\n{response_text}"
            ),
        )
        raw = (response.text or "").strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        recommendations = json.loads(raw)
        if not isinstance(recommendations, list):
            return []

        # Safety-net filter: remove anything that matches basic recommendations
        filtered = []
        for rec in recommendations:
            canonical = rec.get("canonical_form", "").lower().strip()
            if not canonical:
                continue
            is_basic = any(basic in canonical for basic in _BASIC_RECOMMENDATIONS)
            if not is_basic:
                filtered.append(rec)

        return filtered
    except Exception:
        logger.debug("Failed to extract recommendations from response", exc_info=True)
        return []


async def _match_recommendations_to_db(
    recommendations: list[dict], settings: Settings,
) -> list[dict]:
    """Match extracted recommendations against the DB.

    For each recommendation:
    1. Exact match on canonical_form
    2. If no match, Gemini semantic match against all existing canonical forms
    3. If still no match, create a new 'pending' entry

    Enriches each dict with ``db_id``, ``db_status``, ``is_trusted``.
    """
    all_existing = await db.get_all_recommendations()
    existing_map = {r["canonical_form"]: r for r in all_existing}
    existing_canonicals = list(existing_map.keys())

    for rec in recommendations:
        canonical = rec.get("canonical_form", "").strip()

        # 1. Exact match
        db_rec = existing_map.get(canonical)
        if db_rec:
            rec["db_id"] = db_rec["id"]
            rec["db_status"] = db_rec["status"]
            rec["is_trusted"] = db_rec["status"] == "trusted"
            continue

        # 2. Semantic match via Gemini
        if existing_canonicals:
            try:
                client = genai.Client(api_key=settings.gemini_api_key)
                response = await client.aio.models.generate_content(
                    model=settings.gemini_model,
                    contents=(
                        "Given this recommendation:\n"
                        f'"{canonical}"\n\n'
                        "Does it match any of these existing canonical forms? "
                        "A match means they describe the same action even if worded differently.\n\n"
                        + "\n".join(f"- {c}" for c in existing_canonicals)
                        + "\n\nIf there is a match, respond with ONLY the matching canonical form text. "
                        "If there is NO match, respond with exactly: NO_MATCH"
                    ),
                )
                match_text = (response.text or "").strip()
                if match_text != "NO_MATCH" and match_text in existing_map:
                    db_rec = existing_map[match_text]
                    rec["db_id"] = db_rec["id"]
                    rec["db_status"] = db_rec["status"]
                    rec["is_trusted"] = db_rec["status"] == "trusted"
                    rec["canonical_form"] = match_text  # normalize
                    continue
            except Exception:
                logger.debug("Semantic match failed for: %s", canonical, exc_info=True)

        # 3. No match — create new pending entry
        try:
            new_id = await db.create_recommendation(
                canonical, rec.get("category", ""), "pending",
            )
            rec["db_id"] = new_id
            rec["db_status"] = "pending"
            rec["is_trusted"] = False
            # Add to local map so subsequent recs in same batch can match
            new_rec = await db.get_recommendation(new_id)
            if new_rec:
                existing_map[canonical] = new_rec
                existing_canonicals.append(canonical)
        except Exception:
            logger.debug("Failed to create recommendation record for: %s", canonical, exc_info=True)
            rec["db_id"] = None
            rec["db_status"] = "pending"
            rec["is_trusted"] = False

    return recommendations


async def _gate_recommendations(
    response_text: str,
    channel: str,
    thread_ts: str | None,
    say,
    settings: Settings,
    *,
    interaction_id: int | None = None,
    user_name: str = "a user",
) -> tuple[str, str | None]:
    """Orchestrate recommendation approval gating.

    Returns ``(final_text, message_ts_or_None)``.
    If ``message_ts`` is not None, the caller should skip its own ``say()``
    because a redacted message has already been posted.
    """
    # 1. Extract recommendations
    recommendations = await _extract_recommendations(response_text, settings)
    if not recommendations:
        return response_text, None

    # 2. Match against DB
    recommendations = await _match_recommendations_to_db(recommendations, settings)

    # 3. Check if all are trusted
    if all(r.get("is_trusted") for r in recommendations):
        return response_text, None

    # 4. Gate: redact + post + store
    untrusted = [r for r in recommendations if not r.get("is_trusted")]
    redacted_text = redact_recommendations(response_text, untrusted)

    sn_url = settings.sn_instance_url
    linked_redacted = linkify_servicenow_refs(redacted_text, sn_url)
    blocks = format_response_blocks(redacted_text, sn_url)

    say_kwargs: dict = {"text": linked_redacted, "blocks": blocks}
    if thread_ts:
        say_kwargs["thread_ts"] = thread_ts

    msg_response = await say(**say_kwargs)
    posted_ts = msg_response.get("ts", "") if isinstance(msg_response, dict) else ""

    # 5. Store pending approval in DB
    rec_ids = [r["db_id"] for r in untrusted if r.get("db_id")]
    try:
        approval_id = await db.create_pending_rec_approval(
            channel_id=channel,
            message_ts=posted_ts,
            original_text=response_text,
            redacted_text=redacted_text,
            recommendation_ids=rec_ids,
            thread_ts=thread_ts or "",
            interaction_id=interaction_id,
        )
    except Exception:
        logger.warning("Failed to store pending recommendation approval", exc_info=True)
        return response_text, None

    # 6. Post approval request to #it-helpdesk
    if settings.it_helpdesk_channel_id:
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            approval_blocks = format_recommendation_approval_blocks(
                approval_id, untrusted, response_text, user_name,
            )
            resp = await client.chat_postMessage(
                channel=settings.it_helpdesk_channel_id,
                text=f"Recommendation approval needed from {user_name}",
                blocks=approval_blocks,
            )
            await db.update_pending_rec_approval(
                approval_id,
                approval_message_ts=resp["ts"],
                approval_channel=settings.it_helpdesk_channel_id,
            )
        except Exception:
            logger.warning("Failed to post recommendation approval request", exc_info=True)

    return redacted_text, posted_ts


async def _handle_message(event: dict, text: str, say, settings: Settings) -> None:
    """Process a user message through the agent (DM / mention flow)."""
    channel = event["channel"]
    thread_ts = event.get("thread_ts") or event["ts"]
    user_id = event.get("user", "unknown")

    await emit("message_received", {
        "source": "dm", "channel": channel, "user_id": user_id,
        "text": text[:200], "thread_ts": thread_ts,
    })

    # Build conversation key
    conv_key = (channel, thread_ts)

    # Recover history from Slack if we have none (e.g. after bot restart)
    if conv_key not in _conversations:
        recovered = await _recover_thread_history(channel, thread_ts, settings)
        if recovered:
            _conversations[conv_key] = recovered
            history = _conversations[conv_key]
        else:
            _conversations[conv_key] = [{"role": "user", "content": text}]
            history = _conversations[conv_key]
    else:
        history = _conversations[conv_key]
        history.append({"role": "user", "content": text})

    # Create interaction record on first message in this conversation
    if conv_key not in _interaction_ids:
        try:
            requester_name = await _resolve_user_name(user_id, settings)
            iid = await db.create_interaction(
                channel_id=channel, thread_ts=thread_ts, source="dm",
                requester_slack_id=user_id, requester_name=requester_name,
            )
            _interaction_ids[conv_key] = iid
        except Exception:
            logger.debug("Failed to create DM interaction record", exc_info=True)

    # Trim old history
    if len(history) > MAX_HISTORY:
        _conversations[conv_key] = history[-MAX_HISTORY:]
        history = _conversations[conv_key]

    try:
        agent = _get_agent(settings)
        result: AgentResult = await agent.run(
            history,
            user_id=user_id,
            context_articles=list(_ai_context_articles.values()),
        )

        # Append assistant response to history
        history.append({"role": "assistant", "content": result.text})

        await _register_incident_channels(result)

        # Record interaction data
        iid = _interaction_ids.get(conv_key)
        if iid is not None:
            await _record_agent_result(iid, text, result)

        # Recommendation approval gate
        user_display = await _resolve_user_name(user_id, settings)
        final_text, gated_msg_ts = await _gate_recommendations(
            result.text, channel, thread_ts, say, settings,
            interaction_id=iid, user_name=user_display,
        )
        if gated_msg_ts is None:
            sn_url = settings.sn_instance_url
            linked_text = linkify_servicenow_refs(result.text, sn_url)
            blocks = format_response_blocks(result.text, sn_url)
            await say(text=linked_text, blocks=blocks, thread_ts=thread_ts)

        # Post public article feedback buttons + approval requests
        await _post_public_article_followups(result, channel, thread_ts, settings)

    except Exception as exc:
        logger.exception("Error processing message")
        await emit("error", {"source": "dm", "message": str(exc)[:300]})
        blocks = format_error_blocks(
            "Something went wrong processing your request. Please try again."
        )
        await say(text="Error processing request", blocks=blocks, thread_ts=thread_ts)


def _collect_ticket_history(ticket_id: str) -> list[dict]:
    """Collect conversation history from both the #help-it thread and the
    incident channel for a given ticket."""
    messages: list[dict] = []

    # Thread history
    thread_info = _ticket_threads.get(ticket_id)
    if thread_info:
        thread_conv = _conversations.get(thread_info, [])
        messages.extend(thread_conv)

    # Incident channel history
    for ch_id, ctx in _incident_context.items():
        if ctx.get("ticket_id") == ticket_id:
            ch_conv = _conversations.get((ch_id, "incident"), [])
            messages.extend(ch_conv)
            break

    return messages


def _schedule_live_summary(ticket_id: str, settings: Settings) -> None:
    """(Re)schedule a live-summary push to ServiceNow after 5 min of inactivity."""
    existing = _summary_timers.pop(ticket_id, None)
    if existing and not existing.done():
        existing.cancel()

    async def _delayed():
        await asyncio.sleep(_SUMMARY_DEBOUNCE_SECONDS)
        await _push_live_summary(ticket_id, settings)
        _summary_timers.pop(ticket_id, None)

    _summary_timers[ticket_id] = asyncio.create_task(_delayed())


async def _push_live_summary(ticket_id: str, settings: Settings) -> None:
    """Generate an AI summary of all conversations for this ticket and
    write it into the ServiceNow description field."""
    # 1. Collect conversation history from both thread and channel
    history = _collect_ticket_history(ticket_id)
    if not history:
        return

    # 2. Generate summary via Gemini
    summary = await _summarize_conversation(history, settings)

    # 3. Build updated description: original + separator + summary
    original = _ticket_original_descriptions.get(ticket_id, "")
    if not original:
        # Fetch from SN as fallback
        sn = ServiceNowClient(settings.sn_instance_url, settings.sn_username, settings.sn_password)
        try:
            incident = await sn.get_incident(ticket_id)
            if incident:
                desc = incident.get("description", "")
                # Strip any previous live summary
                original = desc.split("\n---------------------------------------------------\n")[0].rstrip()
        finally:
            await sn.close()

    separator = "\n\n---------------------------------------------------\n\n"
    new_description = f"{original}{separator}**Live Summary (auto-updated):**\n{summary}"

    # 4. Update ServiceNow
    sn = ServiceNowClient(settings.sn_instance_url, settings.sn_username, settings.sn_password)
    try:
        incident = await sn.get_incident(ticket_id)
        if incident:
            await sn.update_incident(
                incident["sys_id"],
                {"description": new_description},
                current_state=incident.get("_raw_state", "1"),
            )
            logger.info("Pushed live summary to SN for %s", ticket_id)
    finally:
        await sn.close()


async def _handle_incident_message(
    event: dict, text: str, say, settings: Settings
) -> None:
    """Process a message in an incident channel — no threading, single conversation.

    Seeds the conversation with incident context on first interaction and
    updates the channel's pinned summary message after each agent turn.
    """
    channel = event["channel"]
    user_id = event.get("user", "unknown")

    await emit("message_received", {
        "source": "incident", "channel": channel, "user_id": user_id,
        "text": text[:200], "thread_ts": "",
    })

    # Single conversation per incident channel (no per-thread splitting)
    conv_key = (channel, "incident")
    history = _conversations.setdefault(conv_key, [])

    # Ensure we have incident context — fallback to channel name if needed
    if channel not in _incident_context:
        try:
            slack_client = AsyncWebClient(token=settings.slack_bot_token)
            info = await slack_client.conversations_info(channel=channel)
            ch_name = info.get("channel", {}).get("name", "")
            ticket_id = _ticket_id_from_channel_name(ch_name)
            if ticket_id:
                _incident_context[channel] = {
                    "ticket_id": ticket_id,
                    "title": "",
                    "description": "",
                    "priority": "",
                    "summary_ts": None,
                    "original_text": None,
                    "summary_lines": [],
                }
                logger.info(
                    "Seeded incident context on-the-fly for %s (%s)", ch_name, ticket_id,
                )
        except Exception:
            logger.debug("Could not resolve channel name for %s", channel)

    # Seed context on first interaction so the agent knows what the channel is about
    if not history:
        # Try to recover previous channel messages (e.g. after restart)
        try:
            slack_client = AsyncWebClient(token=settings.slack_bot_token)
            auth = await slack_client.auth_test()
            bot_uid = auth["user_id"]
            resp = await slack_client.conversations_history(channel=channel, limit=MAX_HISTORY)
            msgs = resp.get("messages", [])
            msgs.reverse()  # oldest first
            for msg in msgs:
                if msg.get("subtype"):
                    continue
                msg_text = msg.get("text", "").strip()
                if not msg_text:
                    continue
                if msg.get("user") == bot_uid or msg.get("bot_id"):
                    history.append({"role": "assistant", "content": msg_text})
                else:
                    history.append({"role": "user", "content": msg_text})
            if history:
                logger.info("Recovered %d messages for incident channel %s", len(history), channel)
        except Exception:
            logger.debug("Could not recover history for incident channel %s", channel)

        ctx = _incident_context.get(channel, {})
        if ctx:
            context_msg = (
                f"[Incident context — this channel is dedicated to troubleshooting "
                f"this specific issue]\n"
                f"Ticket: {ctx.get('ticket_id', 'unknown')}\n"
                f"Title: {ctx.get('title', 'N/A')}\n"
                f"Description: {ctx.get('description', 'N/A')}\n"
                f"Priority: {ctx.get('priority', 'N/A')}\n"
                f"Every message in this channel is about this incident. "
                f"When the user asks you to do something, always assume it relates "
                f"to this issue — do not ask for clarification about the topic."
            )
            # Insert context at the beginning, before any recovered messages
            history.insert(0, {"role": "user", "content": context_msg})
            history.insert(1, {
                "role": "assistant",
                "content": (
                    f"Understood. I'm tracking incident {ctx.get('ticket_id', 'this issue')}: "
                    f"\"{ctx.get('title', '')}\". "
                    f"I'll help troubleshoot. What would you like me to do?"
                ),
            })

    history.append({"role": "user", "content": text})

    # Look up or create interaction for this incident channel
    if conv_key not in _interaction_ids:
        ctx = _incident_context.get(channel, {})
        ticket_id = ctx.get("ticket_id")
        if ticket_id:
            try:
                existing = await db.get_interaction_by_ticket(ticket_id)
                if existing:
                    _interaction_ids[conv_key] = existing["id"]
                else:
                    requester_name = await _resolve_user_name(user_id, settings)
                    iid = await db.create_interaction(
                        channel_id=channel, source="incident",
                        requester_slack_id=user_id, requester_name=requester_name,
                        ticket_id=ticket_id, priority=ctx.get("priority", "medium"),
                    )
                    _interaction_ids[conv_key] = iid
            except Exception:
                logger.debug("Failed to look up/create incident interaction", exc_info=True)

    if len(history) > MAX_HISTORY:
        _conversations[conv_key] = history[-MAX_HISTORY:]
        history = _conversations[conv_key]

    try:
        agent = _get_agent(settings)
        result: AgentResult = await agent.run(
            history,
            user_id=user_id,
            context_articles=list(_ai_context_articles.values()),
        )

        history.append({"role": "assistant", "content": result.text})

        await _register_incident_channels(result)

        # Record interaction data
        iid = _interaction_ids.get(conv_key)
        if iid is not None:
            await _record_agent_result(iid, text, result)

        # Recommendation approval gate
        user_display = await _resolve_user_name(user_id, settings)
        final_text, gated_msg_ts = await _gate_recommendations(
            result.text, channel, None, say, settings,
            interaction_id=iid, user_name=user_display,
        )
        if gated_msg_ts is None:
            sn_url = settings.sn_instance_url
            linked_text = linkify_servicenow_refs(result.text, sn_url)
            blocks = format_response_blocks(result.text, sn_url)
            await say(text=linked_text, blocks=blocks)

        # Update the pinned summary message with progress
        await _update_incident_summary(channel, result.text, settings)

        # Schedule debounced live summary push to ServiceNow
        ctx = _incident_context.get(channel, {})
        inc_ticket_id = ctx.get("ticket_id")
        if inc_ticket_id:
            _schedule_live_summary(inc_ticket_id, settings)

        # Post public article feedback buttons + approval requests
        await _post_public_article_followups(result, channel, None, settings)

    except Exception as exc:
        logger.exception("Error processing incident channel message")
        await emit("error", {"source": "incident", "message": str(exc)[:300]})
        blocks = format_error_blocks(
            "Something went wrong processing your request. Please try again."
        )
        await say(text="Error processing request", blocks=blocks)


async def _update_incident_summary(
    channel: str, agent_response: str, settings: Settings
) -> None:
    """Append a summary line to the incident channel's initial message."""
    ctx = _incident_context.get(channel)
    if not ctx or not ctx.get("summary_ts"):
        return

    # Build summary line: first sentence, max 150 chars
    first_sentence = agent_response.split("\n")[0].strip()
    if len(first_sentence) > 150:
        first_sentence = first_sentence[:147] + "..."
    ctx.setdefault("summary_lines", []).append(first_sentence)

    await emit("summary_updated", {
        "channel_id": channel,
        "ticket_id": ctx.get("ticket_id", ""),
        "line": first_sentence,
    })

    # Rebuild the message: original text + live summary
    original = ctx.get("original_text") or ""
    if not original:
        # Fallback: build from context fields
        original = (
            f"*Incident {ctx.get('ticket_id', '')}*\n"
            f"*Title:* {ctx.get('title', '')}\n"
            f"*Priority:* {ctx.get('priority', '')}\n"
            f"*Description:* {ctx.get('description', '')}"
        )
        ctx["original_text"] = original

    summary_bullets = "\n".join(f"• {line}" for line in ctx["summary_lines"])
    updated_text = f"{original}\n\n*LIVE SUMMARY:*\n{summary_bullets}"
    updated_text = linkify_servicenow_refs(updated_text, settings.sn_instance_url)

    try:
        client = AsyncWebClient(token=settings.slack_bot_token)
        await client.chat_update(
            channel=channel,
            ts=ctx["summary_ts"],
            text=updated_text,
        )
    except Exception:
        logger.warning("Failed to update incident summary in %s", channel, exc_info=True)


async def _summarize_conversation(
    history: list[dict], settings: Settings, user_name: str = "the user",
) -> str:
    """Generate a concise summary of a #help-it conversation using Gemini."""
    lines: list[str] = []
    for msg in history:
        role = user_name if msg["role"] == "user" else "IT Agent"
        content = msg["content"]
        # Strip system context prefixes
        if content.startswith("[System"):
            content = content.split("]\n\n", 1)[-1] if "]\n\n" in content else content
        if content.startswith("[System reminder"):
            content = content.split("]\n\n", 1)[-1] if "]\n\n" in content else content
        lines.append(f"{role}: {content}")

    conversation_text = "\n\n".join(lines)

    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=(
                "Summarize the following IT support conversation in 2-4 concise bullet points. "
                f"Refer to the requester by their first name ({user_name}). "
                "Focus on: the issue reported, key troubleshooting steps attempted, "
                "and current status. Use Slack markdown (*bold*, `code`). Be brief.\n\n"
                f"{conversation_text}"
            ),
        )
        return response.text or "No summary available."
    except Exception:
        logger.warning("Failed to generate conversation summary", exc_info=True)
        # Fallback: first user message truncated
        for msg in history:
            if msg["role"] == "user":
                content = msg["content"]
                if content.startswith("[System"):
                    content = content.split("]\n\n", 1)[-1] if "]\n\n" in content else content
                return content[:300]
        return "No summary available."


async def _create_deferred_channel(
    ticket_id: str,
    settings: Settings,
    *,
    thread_info: tuple[str, str] | None = None,
    user_id: str | None = None,
    reason: str = "button",
) -> dict | None:
    """Create a deferred incident channel for a ticket.

    Uses ``_pending_channels`` when available; falls back to fetching the
    ticket from ServiceNow (handles bot restarts).

    Args:
        thread_info: (channel, thread_ts) override when _ticket_threads is empty.
        user_id: Fallback Slack user ID when _pending_channels is empty.
        reason: ``"button"`` or ``"escalation"`` — controls channel content.

    Returns ``{channel_id, channel_name}`` on success or ``None`` on failure.
    """
    pending = _pending_channels.get(ticket_id)
    if pending:
        resolved_user_id = pending["user_id"]
        ticket = pending["ticket"]
    else:
        # Fallback: fetch from ServiceNow (bot may have restarted)
        logger.info("No pending data for %s — fetching from ServiceNow", ticket_id)
        sn_result = await get_ticket(ticket_id, _settings=settings)
        if sn_result.get("error"):
            logger.warning("Could not fetch ticket %s from ServiceNow: %s", ticket_id, sn_result["error"])
            return None
        ticket = sn_result["ticket"]
        resolved_user_id = user_id or "unknown"

    ch = await create_incident_channel(ticket, settings, resolved_user_id)
    if not ch:
        return None

    channel_id = ch["channel_id"]
    channel_name = ch["channel_name"]
    summary_ts = ch.get("summary_ts")

    # Register in global tracking
    _incident_channels.add(channel_id)
    _incident_context[channel_id] = {
        "ticket_id": ticket_id,
        "title": ticket.get("title", ""),
        "description": ticket.get("description", ""),
        "priority": ticket.get("priority", "medium"),
        "summary_ts": summary_ts,
        "original_text": None,
        "summary_lines": [],
    }
    await emit("channel_created", {
        "channel_id": channel_id,
        "channel_name": channel_name,
        "ticket_id": ticket_id,
    })

    slack = AsyncWebClient(token=settings.slack_bot_token)

    # Resolve thread info for permalink
    resolved_thread = thread_info or _ticket_threads.get(ticket_id)
    if resolved_thread:
        help_channel, help_thread_ts = resolved_thread
    else:
        help_channel, help_thread_ts = None, None

    # Recover conversation history
    history: list[dict] = []
    if pending and pending.get("thread_ts") and pending.get("channel"):
        conv_key = (pending["channel"], pending["thread_ts"])
        history = _conversations.get(conv_key, [])
    elif help_channel and help_thread_ts:
        conv_key = (help_channel, help_thread_ts)
        history = _conversations.get(conv_key, [])
        if not history:
            history = await _recover_thread_history(
                help_channel, help_thread_ts, settings,
            )

    # Count real exchanges (exclude system-only messages)
    assistant_count = sum(1 for m in history if m["role"] == "assistant")
    is_early = assistant_count <= 1

    # Resolve the requester's first name for summaries
    requester_first_name = "the user"
    try:
        name_uid = resolved_user_id if resolved_user_id != "unknown" else (user_id or "unknown")
        if name_uid and name_uid != "unknown":
            uinfo = await slack.users_info(user=name_uid)
            profile = uinfo.get("user", {}).get("profile", {})
            full = profile.get("first_name") or profile.get("real_name") or profile.get("display_name") or ""
            if full:
                requester_first_name = full.split()[0]
    except Exception:
        logger.debug("Could not resolve requester name for summary")

    sn_url = settings.sn_instance_url

    # ── Channel content based on reason + conversation stage ──
    if reason == "button" and is_early:
        # Early button press: post the first bot response in the channel
        # Post permalink first
        if help_channel and help_thread_ts:
            try:
                plink = await slack.chat_getPermalink(
                    channel=help_channel, message_ts=help_thread_ts,
                )
                permalink = plink.get("permalink", "")
                if permalink:
                    await slack.chat_postMessage(
                        channel=channel_id,
                        text=f":link: <{permalink}|Original #help-it thread>",
                    )
            except Exception:
                logger.debug("Failed to post permalink in channel %s", channel_id, exc_info=True)

        # Find first assistant response and post it
        for msg in history:
            if msg["role"] == "assistant":
                try:
                    ch_text = _strip_channel_created_lines(msg["content"], channel_name)
                    if ch_text.strip():
                        ch_linked = linkify_servicenow_refs(ch_text, sn_url)
                        ch_blocks = format_response_blocks(ch_text, sn_url)
                        await slack.chat_postMessage(
                            channel=channel_id, text=ch_linked, blocks=ch_blocks,
                        )
                except Exception:
                    logger.debug("Failed to post first response in channel", exc_info=True)
                break
    else:
        # Late button or escalation: generate summary and append to first message
        summary = await _summarize_conversation(history, settings, requester_first_name) if history else ""

        if summary and summary_ts:
            # Build updated first message: ticket info + summary
            sn_link = (
                f"{sn_url}/incident.do"
                f"?sysparm_query=number={ticket.get('ticket_id', ticket_id)}"
            )
            updated_text = (
                f"*Incident {ticket.get('ticket_id', ticket_id)}*\n"
                f"*Title:* {ticket.get('title', '')}\n"
                f"*Priority:* {ticket.get('priority', '')}\n"
                f"*Description:* {ticket.get('description', '')}\n\n"
                f"<{sn_link}|View in ServiceNow>\n\n"
                f"*Conversation Summary:*\n{summary}"
            )
            try:
                await slack.chat_update(
                    channel=channel_id, ts=summary_ts, text=updated_text,
                )
            except Exception:
                logger.debug("Failed to update first message with summary", exc_info=True)
        elif summary:
            # summary_ts unavailable — post summary as a separate message
            await slack.chat_postMessage(
                channel=channel_id,
                text=f"*Conversation Summary:*\n{summary}",
            )

        # Post permalink
        if help_channel and help_thread_ts:
            try:
                plink = await slack.chat_getPermalink(
                    channel=help_channel, message_ts=help_thread_ts,
                )
                permalink = plink.get("permalink", "")
                if permalink:
                    await slack.chat_postMessage(
                        channel=channel_id,
                        text=f":link: <{permalink}|Original #help-it thread>",
                    )
            except Exception:
                logger.debug("Failed to post permalink in channel %s", channel_id, exc_info=True)

    # Clean up pending entry
    _pending_channels.pop(ticket_id, None)

    logger.info("Deferred channel %s created for ticket %s (reason=%s, early=%s)", channel_name, ticket_id, reason, is_early)
    return {"channel_id": channel_id, "channel_name": channel_name}


async def _handle_help_channel_message(
    event: dict, text: str, say, settings: Settings
) -> None:
    """Handle a message in #help-it — auto-creates ticket, then runs agent."""
    channel = event["channel"]
    thread_ts = event.get("thread_ts") or event["ts"]
    user_id = event.get("user", "unknown")

    await emit("message_received", {
        "source": "help-it", "channel": channel, "user_id": user_id,
        "text": text[:200], "thread_ts": thread_ts,
    })

    conv_key = (channel, thread_ts)
    is_new_thread = conv_key not in _conversations and not event.get("thread_ts")

    # ── Auto-create ticket on first message in a new thread (no channel yet) ──
    auto_ticket_info: dict | None = None
    if is_new_thread:
        try:
            ticket_result = await _create_ticket_tool(
                title=text[:120],
                description=text,
                priority="medium",
                create_channel=False,
                _settings=settings,
                _user_id=user_id,
            )
            if ticket_result.get("success"):
                auto_ticket_info = ticket_result
                ticket = ticket_result.get("ticket", {})
                ticket_id = ticket.get("ticket_id", "")

                if ticket_id:
                    _ticket_threads[ticket_id] = (channel, thread_ts)
                    _ticket_original_descriptions[ticket_id] = text
                    # Store for deferred channel creation
                    _pending_channels[ticket_id] = {
                        "user_id": user_id,
                        "ticket": ticket,
                        "thread_ts": thread_ts,
                        "channel": channel,
                    }

                await emit("ticket_created", {
                    "ticket_id": ticket_id,
                    "title": ticket.get("title", ""),
                    "priority": ticket.get("priority", "medium"),
                    "channel_id": "",
                })

                logger.info(
                    "Auto-created ticket %s (no channel) for help-it message",
                    ticket_id,
                )
            else:
                logger.warning("Auto ticket creation failed: %s", ticket_result)
        except Exception:
            logger.exception("Failed to auto-create ticket for help-it message")

    # ── Build / recover conversation history ──
    if conv_key not in _conversations:
        recovered = await _recover_thread_history(channel, thread_ts, settings)
        if recovered:
            _conversations[conv_key] = recovered
            history = _conversations[conv_key]
            # Recover ticket-thread mapping from thread content (survives restarts)
            _recover_ticket_thread_mapping(channel, thread_ts, recovered)
        else:
            history = []
            # Inject ticket context so the agent knows about the auto-created ticket
            if auto_ticket_info:
                ticket = auto_ticket_info.get("ticket", {})
                ticket_id = ticket.get("ticket_id", "")
                context_msg = (
                    f"[System: A ticket {ticket_id} has been created automatically for "
                    f"this request. No private channel has been created yet — the "
                    f"conversation continues in this #help-it thread. Do NOT call "
                    f"create_ticket — it is already done. Do NOT mention a private "
                    f"channel. Focus on diagnosing the issue, searching the KB, and "
                    f"helping the user. Reference {ticket_id} in your response.]\n\n"
                    f"{text}"
                )
                history.append({"role": "user", "content": context_msg})
            else:
                history.append({"role": "user", "content": text})
            _conversations[conv_key] = history
    else:
        history = _conversations[conv_key]
        # Remind the agent about the existing ticket on follow-up messages
        existing_ticket_id = None
        for tid, (ch, ts) in _ticket_threads.items():
            if ch == channel and ts == thread_ts:
                existing_ticket_id = tid
                break
        if existing_ticket_id:
            text_with_ctx = (
                f"[System reminder: Ticket {existing_ticket_id} already exists for "
                f"this thread. Do NOT offer to create a ticket or ask if the user "
                f"wants one — it is already created.]\n\n{text}"
            )
            history.append({"role": "user", "content": text_with_ctx})
        else:
            history.append({"role": "user", "content": text})

    # Create interaction record on first message in this thread
    if conv_key not in _interaction_ids:
        try:
            requester_name = await _resolve_user_name(user_id, settings)
            ticket_id_for_interaction = ""
            priority_for_interaction = "medium"
            if auto_ticket_info:
                t = auto_ticket_info.get("ticket", {})
                ticket_id_for_interaction = t.get("ticket_id", "")
                priority_for_interaction = t.get("priority", "medium")
            iid = await db.create_interaction(
                channel_id=channel, thread_ts=thread_ts, source="help-it",
                requester_slack_id=user_id, requester_name=requester_name,
                ticket_id=ticket_id_for_interaction,
                priority=priority_for_interaction,
            )
            _interaction_ids[conv_key] = iid
        except Exception:
            logger.debug("Failed to create help-it interaction record", exc_info=True)

    if len(history) > MAX_HISTORY:
        _conversations[conv_key] = history[-MAX_HISTORY:]
        history = _conversations[conv_key]

    try:
        agent = _get_agent(settings)
        result: AgentResult = await agent.run(
            history,
            user_id=user_id,
            context_articles=list(_ai_context_articles.values()),
        )

        history.append({"role": "assistant", "content": result.text})

        await _register_incident_channels(result)

        # Record interaction data
        iid = _interaction_ids.get(conv_key)
        if iid is not None:
            await _record_agent_result(iid, text, result)

        # Recommendation approval gate
        user_display = await _resolve_user_name(user_id, settings)
        final_text, gated_msg_ts = await _gate_recommendations(
            result.text, channel, thread_ts, say, settings,
            interaction_id=iid, user_name=user_display,
        )
        if gated_msg_ts is None:
            sn_url = settings.sn_instance_url
            linked_text = linkify_servicenow_refs(result.text, sn_url)
            blocks = format_response_blocks(result.text, sn_url)
            await say(text=linked_text, blocks=blocks, thread_ts=thread_ts)

        # ── Post ticket follow-up with "Move to private channel" button ──
        if auto_ticket_info and auto_ticket_info.get("success"):
            ticket = auto_ticket_info.get("ticket", {})
            ticket_id = ticket.get("ticket_id", "")

            if ticket_id:
                followup_text = (
                    f":ticket: *Ticket {ticket_id} created.* "
                    f"The conversation continues here in this thread. "
                    f"If you need a dedicated private channel, click the button below."
                )
                followup_text = linkify_servicenow_refs(followup_text, sn_url)
                followup_blocks = [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": followup_text,
                        },
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "Move to private channel"},
                                "action_id": "move_to_private_channel",
                                "value": ticket_id,
                                "style": "primary",
                            }
                        ],
                    },
                ]
                await say(
                    text=followup_text,
                    blocks=followup_blocks,
                    thread_ts=thread_ts,
                )

        # Forward thread replies to the incident channel (if one exists)
        if event.get("thread_ts"):
            try:
                inc_channel = _find_incident_channel_for_thread(channel, thread_ts)
                if inc_channel:
                    slack = AsyncWebClient(token=settings.slack_bot_token)
                    try:
                        user_info = await slack.users_info(user=user_id)
                        display_name = (
                            user_info["user"]["profile"].get("display_name")
                            or user_info["user"]["profile"].get("real_name")
                            or user_id
                        )
                    except Exception:
                        display_name = user_id
                    fwd_text = f"*{display_name}* in #help-it thread:\n>{text}"
                    await slack.chat_postMessage(channel=inc_channel, text=fwd_text)
                    await emit("thread_forwarded", {
                        "from_channel": channel, "to_channel": inc_channel,
                        "user": display_name,
                    })
            except Exception:
                logger.debug("Failed to forward thread reply to incident channel", exc_info=True)

        # Collect KB results from this run and store for deferred channel
        kb_results: list[dict] = []
        for tc in result.tool_calls:
            if tc["name"] == "search_knowledge_base":
                kb_results.extend(tc.get("result", {}).get("results", []))

        if kb_results:
            await emit("kb_search", {
                "result_count": len(kb_results),
                "article_ids": [a.get("id", "") for a in kb_results],
            })

        # ── Auto-create channel on escalation (priority bump or assignment) ──
        for tc in result.tool_calls:
            if tc["name"] != "update_ticket":
                continue
            args = tc.get("args", {})
            escalated = args.get("priority") in ("high", "urgent") or args.get("assignee_id")
            if not escalated:
                continue
            tc_ticket_id = args.get("ticket_id", "")
            if not tc_ticket_id:
                continue
            # Check if channel already exists for this ticket
            already_has_channel = any(
                ctx.get("ticket_id") == tc_ticket_id for ctx in _incident_context.values()
            )
            if already_has_channel:
                continue
            logger.info("Escalation detected: auto-creating channel for %s", tc_ticket_id)
            await _unassign_bot_on_escalation(tc_ticket_id, settings)
            ch_result = await _create_deferred_channel(
                tc_ticket_id, settings,
                thread_info=(channel, thread_ts),
                user_id=user_id,
                reason="escalation",
            )
            if ch_result:
                ch_id = ch_result["channel_id"]
                try:
                    slack = AsyncWebClient(token=settings.slack_bot_token)
                    await slack.chat_postMessage(
                        channel=channel,
                        text=(
                            f":rotating_light: Ticket {tc_ticket_id} has been escalated. "
                            f"A private channel <#{ch_id}> has been created for further troubleshooting."
                        ),
                        thread_ts=thread_ts,
                    )
                except Exception:
                    logger.debug("Failed to post escalation channel notice", exc_info=True)

        # ── Keyword fallback: user asked for a human but agent didn't call update_ticket ──
        if _ESCALATION_PATTERNS.search(text):
            # Find the ticket for this thread
            fallback_ticket_id: str | None = None
            for tid, (ch, ts) in _ticket_threads.items():
                if ch == channel and ts == thread_ts:
                    fallback_ticket_id = tid
                    break
            if fallback_ticket_id:
                already_has_channel = any(
                    ctx.get("ticket_id") == fallback_ticket_id
                    for ctx in _incident_context.values()
                )
                if not already_has_channel:
                    logger.info(
                        "Keyword escalation fallback: creating channel for %s",
                        fallback_ticket_id,
                    )
                    await _unassign_bot_on_escalation(fallback_ticket_id, settings)
                    ch_result = await _create_deferred_channel(
                        fallback_ticket_id, settings,
                        thread_info=(channel, thread_ts),
                        user_id=user_id,
                        reason="escalation",
                    )
                    if ch_result:
                        ch_id = ch_result["channel_id"]
                        try:
                            slack = AsyncWebClient(token=settings.slack_bot_token)
                            await slack.chat_postMessage(
                                channel=channel,
                                text=(
                                    f":rotating_light: Ticket {fallback_ticket_id} has been escalated. "
                                    f"A private channel <#{ch_id}> has been created for further troubleshooting."
                                ),
                                thread_ts=thread_ts,
                            )
                        except Exception:
                            logger.debug("Failed to post keyword escalation notice", exc_info=True)

        # Handle any additional tickets the agent may have created (shouldn't happen
        # but keep as safety net)
        for tc in result.tool_calls:
            if tc["name"] != "create_ticket":
                continue
            r = tc.get("result", {})
            if not r.get("success"):
                continue
            ticket = r.get("ticket", {})
            tid = ticket.get("ticket_id", "")
            if tid and tid not in _ticket_threads:
                _ticket_threads[tid] = (channel, thread_ts)

        # Post public article feedback buttons + approval requests
        await _post_public_article_followups(result, channel, thread_ts, settings)

        # Schedule debounced live summary push to ServiceNow
        help_ticket_id: str | None = None
        for tid, (ch, ts) in _ticket_threads.items():
            if ch == channel and ts == thread_ts:
                help_ticket_id = tid
                break
        if help_ticket_id:
            _schedule_live_summary(help_ticket_id, settings)

    except Exception as exc:
        logger.exception("Error processing #help-it message")
        await emit("error", {"source": "help-it", "message": str(exc)[:300]})
        blocks = format_error_blocks(
            "Something went wrong processing your request. Please try again."
        )
        await say(text="Error processing request", blocks=blocks, thread_ts=thread_ts)


async def _post_public_article_followups(
    result: AgentResult,
    channel: str,
    thread_ts: str | None,
    settings: Settings,
) -> None:
    """After the agent responds, check for public article results and post feedback buttons + approvals."""
    for tc in result.tool_calls:
        if tc["name"] != "search_public_articles":
            continue
        r = tc.get("result", {})

        # Post feedback buttons for returned articles
        articles = r.get("results", [])
        if articles:
            blocks = format_public_article_blocks(articles)
            if blocks:
                try:
                    client = AsyncWebClient(token=settings.slack_bot_token)
                    kwargs: dict = {"channel": channel, "text": "Public articles", "blocks": blocks}
                    if thread_ts:
                        kwargs["thread_ts"] = thread_ts
                    await client.chat_postMessage(**kwargs)
                except Exception:
                    logger.debug("Failed to post public article blocks", exc_info=True)

        # Post approval requests for pending articles
        pending = r.get("needs_approval", [])
        if pending and settings.it_helpdesk_channel_id:
            client = AsyncWebClient(token=settings.slack_bot_token)
            for item in pending:
                article = await db.get_public_article(item["article_id"])
                if not article:
                    continue
                blocks = format_approval_blocks(article)
                try:
                    resp = await client.chat_postMessage(
                        channel=settings.it_helpdesk_channel_id,
                        text=f"Article approval needed: {article['title']}",
                        blocks=blocks,
                    )
                    # Store the approval message TS for timeout tracking
                    await db.update_public_article(
                        article["id"],
                        approval_message_ts=resp["ts"],
                        approval_channel=settings.it_helpdesk_channel_id,
                    )
                except Exception:
                    logger.debug(
                        "Failed to post approval request for article %d", article["id"],
                        exc_info=True,
                    )

        await emit("public_article_search", {
            "result_count": len(articles),
            "pending_count": len(pending),
        })


async def post_resolution_update(
    ticket_id: str, ticket_data: dict, settings: Settings
) -> None:
    """Post a resolution message to the original #help-it thread and rename the incident channel."""
    client = AsyncWebClient(token=settings.slack_bot_token)

    # Post to the original #help-it thread
    thread_info = _ticket_threads.pop(ticket_id, None)
    if thread_info is not None:
        channel, thread_ts = thread_info
        status = ticket_data.get("status", "resolved")
        title = ticket_data.get("title", ticket_id)
        close_notes = ticket_data.get("close_notes", "")

        message = f":white_check_mark: *Ticket {ticket_id} — {status}.* {title}"
        if close_notes:
            message += f"\n>_{close_notes}_"
        message = linkify_servicenow_refs(message, settings.sn_instance_url)

        await client.chat_postMessage(channel=channel, text=message, thread_ts=thread_ts)

    await emit("ticket_resolved", {
        "ticket_id": ticket_id,
        "close_notes": ticket_data.get("close_notes", "")[:200],
    })

    # Update interaction record: resolved
    try:
        interaction = await db.get_interaction_by_ticket(ticket_id)
        if interaction:
            resolved_by_bot = 1 if not interaction.get("assignee_slack_id") else 0
            await db.update_interaction(
                interaction["id"],
                status="resolved",
                resolved_at=db._now_iso(),
                close_notes=ticket_data.get("close_notes", ""),
                resolved_by_bot=resolved_by_bot,
            )
    except Exception:
        logger.debug("Failed to update interaction on resolution for %s", ticket_id, exc_info=True)

    # Rename the incident channel to append "-resolved"
    await _rename_incident_channel_resolved(ticket_id, client)

    # Schedule auto-close after 48 hours
    _resolved_pending_close[ticket_id] = time.time()
    logger.info("Ticket %s queued for auto-close in 48h", ticket_id)

    # Post resolution notice with "Archive Now" button in the incident channel
    channel_id: str | None = None
    for ch_id, ctx in _incident_context.items():
        if ctx.get("ticket_id") == ticket_id:
            channel_id = ch_id
            break

    if channel_id is not None:
        notice = (
            f":white_check_mark: *Ticket {ticket_id} has been resolved.*\n\n"
            f"This channel will be automatically archived in 48 hours. "
            f"If you'd like to archive it sooner, click the button below."
        )
        notice = linkify_servicenow_refs(notice, settings.sn_instance_url)
        try:
            await client.chat_postMessage(
                channel=channel_id,
                text=notice,
                blocks=[
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": notice},
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "Archive Now"},
                                "style": "danger",
                                "action_id": "archive_channel_now",
                                "value": ticket_id,
                            }
                        ],
                    },
                ],
            )
        except Exception:
            logger.debug("Failed to post resolution notice in %s", channel_id, exc_info=True)


async def _handle_ticket_assigned(
    ticket_id: str, assignee_sn_sys_id: str, settings: Settings
) -> None:
    """Invite the assigned user to the incident Slack channel and post a notification.

    If no channel exists yet (deferred flow), auto-creates one first.
    """
    await _unassign_bot_on_escalation(ticket_id, settings)

    # Reverse-lookup: find channel_id whose context has this ticket_id
    channel_id: str | None = None
    for ch_id, ctx in _incident_context.items():
        if ctx.get("ticket_id") == ticket_id:
            channel_id = ch_id
            break

    # Auto-create channel if none exists yet (deferred flow / escalation)
    if channel_id is None:
        logger.info("Assignment: auto-creating channel for ticket %s", ticket_id)
        t_info = _ticket_threads.get(ticket_id)
        ch_result = await _create_deferred_channel(
            ticket_id, settings,
            thread_info=t_info,
            reason="escalation",
        )
        if ch_result:
            channel_id = ch_result["channel_id"]
            # Notify the #help-it thread about escalation
            if t_info:
                try:
                    slack = AsyncWebClient(token=settings.slack_bot_token)
                    await slack.chat_postMessage(
                        channel=t_info[0],
                        text=(
                            f":bust_in_silhouette: Ticket {ticket_id} has been assigned to a Support Agent. "
                            f"A private channel <#{channel_id}> has been created for further troubleshooting."
                        ),
                        thread_ts=t_info[1],
                    )
                except Exception:
                    logger.debug("Failed to post escalation notice in #help-it thread", exc_info=True)

    if channel_id is None:
        logger.debug("No incident channel found for ticket %s — skipping invite", ticket_id)
        return

    # Resolve ServiceNow sys_id → Slack user
    slack_user = await resolve_sn_user_to_slack(assignee_sn_sys_id, settings)
    if slack_user is None:
        logger.info(
            "Could not resolve SN user %s to Slack — skipping invite for %s",
            assignee_sn_sys_id, ticket_id,
        )
        return

    slack_id = slack_user["slack_id"]
    real_name = slack_user["real_name"]
    client = AsyncWebClient(token=settings.slack_bot_token)

    await emit("ticket_assigned", {
        "ticket_id": ticket_id, "assignee_name": real_name, "channel_id": channel_id,
    })

    # Update interaction record with assignee info
    try:
        interaction = await db.get_interaction_by_ticket(ticket_id)
        if interaction:
            await db.update_interaction(
                interaction["id"],
                assignee_slack_id=slack_id,
                assignee_name=real_name,
                status="in_progress",
            )
    except Exception:
        logger.debug("Failed to update interaction assignee for %s", ticket_id, exc_info=True)

    # Invite the assignee to the channel
    try:
        await client.conversations_invite(channel=channel_id, users=slack_id)
        logger.info("Invited %s (%s) to incident channel %s", real_name, slack_id, channel_id)
    except Exception as exc:
        # Handle "already_in_channel" gracefully
        if "already_in_channel" in str(exc):
            logger.debug("%s is already in channel %s", slack_id, channel_id)
        else:
            logger.warning(
                "Failed to invite %s to channel %s", slack_id, channel_id, exc_info=True,
            )

    # Post assignment notification
    try:
        await client.chat_postMessage(
            channel=channel_id,
            text=f":bust_in_silhouette: *{real_name}* has been assigned to this incident.",
        )
    except Exception:
        logger.warning(
            "Failed to post assignment message in %s", channel_id, exc_info=True,
        )


async def _rename_incident_channel_resolved(
    ticket_id: str, client: AsyncWebClient
) -> None:
    """Find the incident channel for *ticket_id* and append '-resolved' to its name."""
    # Reverse-lookup: find channel_id whose context has this ticket_id
    channel_id: str | None = None
    for ch_id, ctx in _incident_context.items():
        if ctx.get("ticket_id") == ticket_id:
            channel_id = ch_id
            break

    if channel_id is None:
        return

    try:
        info = await client.conversations_info(channel=channel_id)
        current_name = info["channel"]["name"]
        if current_name.endswith("-resolved"):
            return  # already renamed
        new_name = f"{current_name}-resolved"[:80]
        await client.conversations_rename(channel=channel_id, name=new_name)
        logger.info("Renamed incident channel %s → %s", current_name, new_name)
    except Exception:
        logger.warning("Failed to rename incident channel %s", channel_id, exc_info=True)


async def start_auto_close_loop(settings: Settings) -> None:
    """Background loop that closes resolved tickets after 48 hours and archives their channels."""
    while True:
        try:
            await _process_pending_auto_closes(settings)
        except Exception:
            logger.warning("Auto-close loop iteration failed", exc_info=True)
        await asyncio.sleep(_AUTO_CLOSE_CHECK_INTERVAL)


async def _process_pending_auto_closes(settings: Settings) -> None:
    """Check for tickets resolved > 48h ago and close them."""
    now = time.time()
    ready = [
        tid for tid, resolved_at in _resolved_pending_close.items()
        if now - resolved_at >= _AUTO_CLOSE_DELAY
    ]
    if not ready:
        return

    logger.info("Auto-closing %d ticket(s) after 48h: %s", len(ready), ", ".join(ready))

    client = AsyncWebClient(token=settings.slack_bot_token)
    sn_client = ServiceNowClient(
        settings.sn_instance_url, settings.sn_username, settings.sn_password
    )

    try:
        for ticket_id in ready:
            await _auto_close_ticket(ticket_id, settings, client, sn_client)
            _resolved_pending_close.pop(ticket_id, None)
    finally:
        await sn_client.close()


async def _auto_close_ticket(
    ticket_id: str,
    settings: Settings,
    slack_client: AsyncWebClient,
    sn_client: ServiceNowClient,
) -> None:
    """Close a single ticket in ServiceNow, post a note, and archive the channel."""
    # 1. Close ticket in ServiceNow
    try:
        incident = await sn_client.get_incident(ticket_id)
        if incident is None:
            logger.warning("Auto-close: ticket %s not found in ServiceNow", ticket_id)
            return
        # Only close if still in resolved state
        if incident.get("status") not in ("resolved",):
            logger.info("Auto-close: ticket %s is %s, skipping", ticket_id, incident.get("status"))
            return
        await sn_client.update_incident(
            incident["sys_id"],
            {"status": "closed", "close_notes": "Auto-closed after 48 hours in resolved state."},
            current_state=incident.get("_raw_state", "6"),
        )
        logger.info("Auto-closed ticket %s in ServiceNow", ticket_id)
    except Exception:
        logger.warning("Auto-close: failed to close %s in ServiceNow", ticket_id, exc_info=True)
        return  # don't archive if close failed

    # 2. Find the incident channel
    channel_id: str | None = None
    for ch_id, ctx in _incident_context.items():
        if ctx.get("ticket_id") == ticket_id:
            channel_id = ch_id
            break

    if channel_id is None:
        return

    # 3. Post farewell note
    sn_url = settings.sn_instance_url
    note = (
        f":lock: *This channel is now being archived.*\n\n"
        f"Ticket {ticket_id} was automatically closed after 48 hours in "
        f"resolved state. If you need further assistance on this issue, "
        f"please open a new request in #help-it."
    )
    note = linkify_servicenow_refs(note, sn_url)
    try:
        await slack_client.chat_postMessage(channel=channel_id, text=note)
    except Exception:
        logger.warning("Auto-close: failed to post farewell note in %s", channel_id, exc_info=True)

    # 4. Archive the channel
    try:
        await slack_client.conversations_archive(channel=channel_id)
        logger.info("Archived incident channel %s for ticket %s", channel_id, ticket_id)
    except Exception:
        logger.warning("Auto-close: failed to archive channel %s", channel_id, exc_info=True)

    await emit("ticket_auto_closed", {"ticket_id": ticket_id, "channel_id": channel_id})
    await emit("channel_archived", {"ticket_id": ticket_id, "channel_id": channel_id})

    # Update interaction record: closed
    try:
        interaction = await db.get_interaction_by_ticket(ticket_id)
        if interaction:
            await db.update_interaction(interaction["id"], status="closed")
    except Exception:
        logger.debug("Failed to update interaction on auto-close for %s", ticket_id, exc_info=True)

    # Clean up tracking
    _incident_channels.discard(channel_id)
    _incident_context.pop(channel_id, None)


# ---------------------------------------------------------------------------
# Approval timeout loop
# ---------------------------------------------------------------------------

_APPROVAL_TIMEOUT_MINUTES = 30
_APPROVAL_CHECK_INTERVAL = 5 * 60  # 5 minutes


async def start_approval_timeout_loop(settings: Settings) -> None:
    """Background loop that auto-denies article approvals older than 30 minutes."""
    while True:
        await asyncio.sleep(_APPROVAL_CHECK_INTERVAL)
        try:
            await _process_expired_approvals(settings)
        except Exception:
            logger.warning("Approval timeout loop iteration failed", exc_info=True)


async def _process_expired_approvals(settings: Settings) -> None:
    """Auto-deny articles whose approval request is older than the timeout."""
    expired = await db.get_pending_approvals(older_than_minutes=_APPROVAL_TIMEOUT_MINUTES)
    if not expired:
        return

    logger.info("Auto-denying %d expired article approval(s)", len(expired))
    client = AsyncWebClient(token=settings.slack_bot_token)

    for article in expired:
        article_id = article["id"]
        await db.update_public_article(article_id, status="denied")

        # Update the Slack approval message if we have the TS
        msg_ts = article.get("approval_message_ts")
        ch = article.get("approval_channel")
        if msg_ts and ch:
            try:
                title = article.get("title", f"Article {article_id}")
                await client.chat_update(
                    channel=ch,
                    ts=msg_ts,
                    text=f":hourglass: *{title}* — Auto-denied (timed out after {_APPROVAL_TIMEOUT_MINUTES}min)",
                    blocks=[{
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": (
                                f":hourglass: *{title}* — Auto-denied "
                                f"(timed out after {_APPROVAL_TIMEOUT_MINUTES}min)"
                            ),
                        },
                    }],
                )
            except Exception:
                logger.debug("Failed to update expired approval message for article %d", article_id, exc_info=True)

        await emit("article_auto_denied", {"article_id": article_id})


# ---------------------------------------------------------------------------
# Recommendation approval timeout loop
# ---------------------------------------------------------------------------

async def start_recommendation_timeout_loop(settings: Settings) -> None:
    """Background loop that auto-denies recommendation approvals older than 30 minutes."""
    while True:
        await asyncio.sleep(_APPROVAL_CHECK_INTERVAL)
        try:
            await _process_expired_rec_approvals(settings)
        except Exception:
            logger.warning("Recommendation timeout loop iteration failed", exc_info=True)


async def _process_expired_rec_approvals(settings: Settings) -> None:
    """Auto-deny recommendation approvals older than the timeout."""
    expired = await db.get_expired_rec_approvals(older_than_minutes=_APPROVAL_TIMEOUT_MINUTES)
    if not expired:
        return

    logger.info("Auto-denying %d expired recommendation approval(s)", len(expired))
    client = AsyncWebClient(token=settings.slack_bot_token)

    for approval in expired:
        approval_id = approval["id"]
        await db.update_pending_rec_approval(approval_id, status="denied")

        # Increment denial counts
        rec_ids = json.loads(approval.get("recommendation_ids", "[]"))
        for rec_id in rec_ids:
            await db.increment_recommendation_denial(rec_id)

        # Update the user's message with timeout notice
        try:
            timeout_text = (
                f"{approval['redacted_text'].rsplit(':hourglass_flowing_sand:', 1)[0].rstrip()}"
                "\n\n:hourglass: _The specific troubleshooting steps were not reviewed in time "
                "and have been removed. Please contact the help desk for further assistance._"
            )
            blocks = format_response_blocks(timeout_text, settings.sn_instance_url)
            await client.chat_update(
                channel=approval["channel_id"],
                ts=approval["message_ts"],
                text=timeout_text,
                blocks=blocks,
            )
        except Exception:
            logger.debug(
                "Failed to update user message for expired recommendation %d",
                approval_id, exc_info=True,
            )

        # Update the #it-helpdesk message
        msg_ts = approval.get("approval_message_ts")
        ch = approval.get("approval_channel")
        if msg_ts and ch:
            try:
                await client.chat_update(
                    channel=ch,
                    ts=msg_ts,
                    text=f":hourglass: Recommendations auto-denied (timed out after {_APPROVAL_TIMEOUT_MINUTES}min)",
                    blocks=[{
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": (
                                f":hourglass: Recommendations auto-denied "
                                f"(timed out after {_APPROVAL_TIMEOUT_MINUTES}min)"
                            ),
                        },
                    }],
                )
            except Exception:
                logger.debug(
                    "Failed to update expired recommendation approval message %d",
                    approval_id, exc_info=True,
                )

        await emit("recommendation_auto_denied", {"approval_id": approval_id})
