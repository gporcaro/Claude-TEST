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
from it_agent.agent.core import Agent, AgentResult, REFINEMENT_SYSTEM_PROMPT
from it_agent.bot.events import emit
from it_agent.bot.formatters import (
    format_approval_blocks,
    format_collaborative_context_blocks,
    format_collaborative_review_blocks,
    format_debug_blocks,
    format_error_blocks,
    format_kb_suggestion_blocks,
    format_public_article_blocks,
    format_recommendation_approval_blocks,
    format_refinement_context_blocks,
    format_refinement_send_button,
    format_response_blocks,
    linkify_servicenow_refs,
    redact_recommendations,
)
from it_agent.config import Settings
from it_agent import db
from it_agent.servicenow.client import ServiceNowClient
from it_agent.kb.indexer import _strip_html
from it_agent.tools.tickets import create_incident_channel, create_ticket as _create_ticket_tool, get_ticket, update_ticket
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

# Debug channel ID resolved at startup (None = not found / disabled)
_debug_channel_id: str | None = None

# Active refinement threads: (helpdesk_channel, approval_message_ts) → approval_id
_active_refinements: dict[tuple[str, str], int] = {}

# Deduplication: track message timestamps currently being processed to prevent
# the same Slack event from being handled concurrently (e.g. Socket Mode retries
# that arrive while an earlier delivery is still awaiting an API call).
_processing_messages: set[str] = set()


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


async def resolve_debug_channel(settings: Settings) -> None:
    """Find the debug channel by name and cache its ID."""
    global _debug_channel_id
    if not settings.debug_channel_name:
        logger.info("Debug channel name is empty — debug logging disabled")
        return

    try:
        client = AsyncWebClient(token=settings.slack_bot_token)
        cursor = None
        while True:
            kwargs: dict = {"types": "public_channel,private_channel", "limit": 200}
            if cursor:
                kwargs["cursor"] = cursor
            resp = await client.conversations_list(**kwargs)
            for ch in resp.get("channels", []):
                if ch.get("name") == settings.debug_channel_name:
                    _debug_channel_id = ch["id"]
                    logger.info(
                        "Debug channel resolved: #%s → %s",
                        settings.debug_channel_name, _debug_channel_id,
                    )
                    return
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        logger.warning(
            "Debug channel #%s not found — debug logging disabled",
            settings.debug_channel_name,
        )
    except Exception:
        logger.warning("Failed to resolve debug channel", exc_info=True)


async def recover_active_refinements(settings: Settings) -> None:
    """Re-populate _active_refinements from DB for approvals in 'refining' status.

    This ensures refinement threads survive bot restarts.
    """
    try:
        refining = await db.get_refining_rec_approvals()
        for approval in refining:
            ch = approval.get("approval_channel")
            ts = approval.get("approval_message_ts")
            if not ch or not ts:
                continue
            approval_id = approval["id"]
            _active_refinements[(ch, ts)] = approval_id
            # Seed conversation context so the agent can continue refining
            _conversations[(ch, ts)] = [
                {
                    "role": "assistant",
                    "content": (
                        f"Here is the original response that needs refinement:\n\n"
                        f"{approval['original_text']}\n\n"
                        f"I'm ready to help you refine this response. "
                        f"What changes would you like to make?"
                    ),
                },
            ]
        if refining:
            logger.info(
                "Recovered %d active refinement(s) from database", len(refining),
            )
    except Exception:
        logger.warning("Failed to recover active refinements", exc_info=True)


def _summarize_args(name: str, args: dict | None) -> str:
    """One-liner summary of tool arguments (truncated)."""
    if not args:
        return ""
    if name == "search_knowledge_base":
        return args.get("query", "")[:80]
    if name == "search_public_articles":
        return args.get("query", "")[:80]
    parts = ", ".join(f"{k}={v}" for k, v in args.items() if v is not None)
    return parts[:120]


def _summarize_result(name: str, res) -> str:
    """One-liner summary of a tool result (truncated)."""
    if res is None:
        return "no result"
    if isinstance(res, dict):
        if name in ("search_knowledge_base", "search_public_articles"):
            results = res.get("results", [])
            return f"{len(results)} result(s)"
        if res.get("success") is not None:
            return "success" if res["success"] else "failed"
        if res.get("error"):
            return f"error: {str(res['error'])[:80]}"
    text = str(res)
    return text[:120] if len(text) > 120 else text


async def _post_debug_summary(
    source: str,
    user_id: str,
    user_message: str,
    result: AgentResult,
    channel: str,
    thread_ts: str | None,
    settings: Settings,
    incident_channel_id: str = "",
    ticket_id: str = "",
) -> None:
    """Post a structured reasoning trace to the debug channel (fire-and-forget)."""
    if not _debug_channel_id:
        return
    try:
        client = AsyncWebClient(token=settings.slack_bot_token)

        # Build thread permalink
        thread_link = ""
        if thread_ts:
            try:
                link_resp = await client.chat_getPermalink(channel=channel, message_ts=thread_ts)
                thread_link = link_resp.get("permalink", "")
            except Exception:
                pass

        # Build numbered steps
        steps: list[str] = []
        step = 1

        # Step 1: AI Context articles
        article_titles = [a["title"] for a in _ai_context_articles.values()]
        # Simple keyword relevance: check if any words from the user message appear in article titles
        relevant = []
        msg_words = set(user_message.lower().split())
        for title in article_titles:
            title_words = set(title.lower().split())
            if msg_words & title_words:
                relevant.append(title.replace("[AI Context] ", ""))
        relevance = f" — likely relevant: {', '.join(relevant)}" if relevant else ""
        steps.append(f"#{step} [AI Context] {len(article_titles)} article(s) loaded{relevance}")
        step += 1

        # Steps for each tool call
        for tc in result.tool_calls:
            name = tc["name"]
            args = tc.get("args")
            res = tc.get("result")
            if name in ("search_knowledge_base", "search_public_articles"):
                query = _summarize_args(name, args)
                hits = _summarize_result(name, res)
                steps.append(f'#{step} :mag: KB search: "{query}" — {hits}')
            else:
                arg_str = _summarize_args(name, args)
                res_str = _summarize_result(name, res)
                emoji = ":white_check_mark:" if "success" in res_str else ":gear:"
                steps.append(f"#{step} {emoji} {name}({arg_str}) → {res_str}")
            step += 1

        # Final step: response preview
        preview = result.text[:200]
        if len(result.text) > 200:
            preview += "..."
        steps.append(f"#{step} :speech_balloon: Response: {preview}")

        blocks = format_debug_blocks(
            source=source,
            user_id=user_id,
            user_message=user_message,
            steps=steps,
            thread_link=thread_link,
            incident_channel_id=incident_channel_id,
            ticket_id=ticket_id,
        )

        await client.chat_postMessage(
            channel=_debug_channel_id,
            text=f"Debug: {source} interaction from <@{user_id}>",
            blocks=blocks,
        )
    except Exception:
        logger.debug("Failed to post debug summary", exc_info=True)


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

        # Deduplicate (same guard as handle_message)
        msg_ts = event.get("ts", "")
        if msg_ts in _processing_messages:
            return
        _processing_messages.add(msg_ts)

        try:
            # Strip the bot mention from the text
            text = re.sub(r"<@[A-Z0-9]+>\s*", "", event.get("text", "")).strip()
            if not text:
                await say("Hi! I'm the IT Support Agent. How can I help you?")
                return
            await _handle_message(event, text, say, settings)
        finally:
            _processing_messages.discard(msg_ts)

    @app.event("message")
    async def handle_message(event: dict, say) -> None:
        """Route messages: DMs → agent, #help-it → proactive handler, others → ignore."""
        # Ignore bot messages, edits, etc.
        if event.get("subtype") or event.get("bot_id"):
            return

        # Deduplicate: prevent the same event from being handled concurrently
        # (Socket Mode can re-deliver while we're awaiting an API call)
        msg_ts = event.get("ts", "")
        if msg_ts in _processing_messages:
            return
        _processing_messages.add(msg_ts)

        text = event.get("text", "").strip()
        if not text:
            _processing_messages.discard(msg_ts)
            return

        channel_type = event.get("channel_type", "")
        channel = event.get("channel", "")

        try:
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

            # Fallback: private channel not yet registered (e.g. created
            # right before a bot restart and missed by discovery).  Look up
            # the channel name and, if it matches the incident pattern,
            # register it on the fly so the conversation can continue.
            if channel_type == "group" and channel not in _incident_channels:
                try:
                    _client = AsyncWebClient(token=settings.slack_bot_token)
                    _info = await _client.conversations_info(channel=channel)
                    _ch_name = _info.get("channel", {}).get("name", "")
                    _tid = _ticket_id_from_channel_name(_ch_name)
                    if _tid:
                        _incident_channels.add(channel)
                        if channel not in _incident_context:
                            _incident_context[channel] = {
                                "ticket_id": _tid,
                                "title": "",
                                "description": "",
                                "priority": "",
                                "summary_ts": None,
                                "original_text": None,
                                "summary_lines": [],
                            }
                        logger.info(
                            "Late-registered incident channel %s (%s) for %s",
                            channel, _ch_name, _tid,
                        )
                        await _handle_incident_message(event, text, say, settings)
                        return
                except Exception:
                    logger.debug("Failed late-discovery for channel %s", channel, exc_info=True)

            # Refinement threads in #it-helpdesk
            thread_ts = event.get("thread_ts", "")
            if (
                settings.it_helpdesk_channel_id
                and channel == settings.it_helpdesk_channel_id
                and thread_ts
                and (channel, thread_ts) in _active_refinements
            ):
                await _handle_refinement_message(event, text, settings)
                return

            # Other channels → ignore (handled by app_mention only)
        finally:
            _processing_messages.discard(msg_ts)

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

    async def _build_rec_decision_line(
        emoji: str, action: str, actor_name: str,
        rec_ids: list[int], approval: dict, settings: Settings,
    ) -> str:
        """Build a single-line summary for #it-helpdesk after approve/deny.

        Format: <emoji> Recommendations <action> by <actor> — <short summary> | <link>
        """
        # Build short summary from recommendation canonical forms
        summaries: list[str] = []
        for rid in rec_ids:
            rec = await db.get_recommendation(rid)
            if rec:
                summaries.append(rec["canonical_form"])
        summary_text = ", ".join(summaries) if summaries else "N/A"
        # Truncate if too long
        if len(summary_text) > 120:
            summary_text = summary_text[:117] + "..."

        # Build permalink to the original thread/channel
        link_text = ""
        conv_channel = approval.get("channel_id", "")
        conv_thread = approval.get("thread_ts", "")
        if conv_channel:
            try:
                client = AsyncWebClient(token=settings.slack_bot_token)
                msg_ts = conv_thread or approval.get("message_ts", "")
                if msg_ts:
                    link_resp = await client.chat_getPermalink(
                        channel=conv_channel, message_ts=msg_ts,
                    )
                    permalink = link_resp.get("permalink", "")
                    if permalink:
                        link_text = f" | <{permalink}|View thread>"
            except Exception:
                logger.debug("Failed to get permalink for rec decision line", exc_info=True)

        return (
            f"{emoji} Recommendations {action} by {actor_name}"
            f" — _{summary_text}_{link_text}"
        )

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

        # Check if conversation moved to an incident channel
        orig_channel = approval["channel_id"]
        orig_thread = approval.get("thread_ts", "")
        incident_ch = _find_incident_channel_for_thread(orig_channel, orig_thread)

        # Deliver the approved response
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            sn_url = settings.sn_instance_url
            original = approval["original_text"]
            linked = linkify_servicenow_refs(original, sn_url)
            blocks = format_response_blocks(original, sn_url)

            if incident_ch:
                # Post as a new message in the incident channel
                await client.chat_postMessage(
                    channel=incident_ch,
                    text=linked,
                    blocks=blocks,
                )
                # Update live summary with the approved recommendation
                await _update_incident_summary(incident_ch, original, settings)
                inc_ctx = _incident_context.get(incident_ch, {})
                inc_tid = inc_ctx.get("ticket_id")
                if inc_tid:
                    _push_live_summary_now(inc_tid, settings)
            else:
                # Update the redacted message in the original thread
                await client.chat_update(
                    channel=orig_channel,
                    ts=approval["message_ts"],
                    text=linked,
                    blocks=blocks,
                )
                # Post a thread reply so the user gets a Slack notification
                if orig_thread:
                    await client.chat_postMessage(
                        channel=orig_channel,
                        thread_ts=orig_thread,
                        text=(
                            ":white_check_mark: The troubleshooting steps for this issue "
                            "have been reviewed and approved by IT. "
                            "Please see the updated message above."
                        ),
                    )
        except Exception:
            logger.debug("Failed to update user message on recommendation approval", exc_info=True)

        # Update the #it-helpdesk approval message with summary + link
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            ch = approval.get("approval_channel") or body["channel"]["id"]
            ts = approval.get("approval_message_ts") or body["message"]["ts"]
            summary_line = await _build_rec_decision_line(
                ":white_check_mark:", "approved", approver_name,
                rec_ids, approval, settings,
            )
            await client.chat_update(
                channel=ch,
                ts=ts,
                text=summary_line,
                blocks=[{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": summary_line},
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

        # Escalate the ticket to high priority
        escalated_ticket_id = ""
        interaction_id = approval.get("interaction_id")
        if interaction_id:
            try:
                interaction = await db.get_interaction(interaction_id)
                if interaction and interaction.get("ticket_id"):
                    escalated_ticket_id = interaction["ticket_id"]
                    await update_ticket(
                        escalated_ticket_id,
                        priority="high",
                        comment="Recommendation denied by IT review — escalating for agent follow-up.",
                        _settings=settings,
                    )
                    logger.info("Escalated ticket %s after recommendation denial", escalated_ticket_id)
            except Exception:
                logger.debug("Failed to escalate ticket on recommendation denial", exc_info=True)

        # Update the user's message with escalation notice
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            denial_text = (
                f"{approval['redacted_text'].rsplit(':hourglass_flowing_sand:', 1)[0].rstrip()}"
                "\n\n:rotating_light: _The troubleshooting steps have been reviewed by IT. "
                "This issue has been escalated and a Support Agent will follow up with you._"
            )
            blocks = format_response_blocks(denial_text, settings.sn_instance_url)
            await client.chat_update(
                channel=approval["channel_id"],
                ts=approval["message_ts"],
                text=denial_text,
                blocks=blocks,
            )
            # Post a thread reply so the user gets a Slack notification
            thread = approval.get("thread_ts", "")
            if thread:
                await client.chat_postMessage(
                    channel=approval["channel_id"],
                    thread_ts=thread,
                    text=(
                        ":rotating_light: The troubleshooting steps for this issue have been "
                        "reviewed by IT and escalated for further assistance. "
                        "A Support Agent will follow up with you."
                    ),
                )
        except Exception:
            logger.debug("Failed to update user message on recommendation denial", exc_info=True)

        # Update the #it-helpdesk message with summary + link
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            ch = approval.get("approval_channel") or body["channel"]["id"]
            ts = approval.get("approval_message_ts") or body["message"]["ts"]
            summary_line = await _build_rec_decision_line(
                ":no_entry:", "denied", denier_name,
                rec_ids, approval, settings,
            )
            await client.chat_update(
                channel=ch,
                ts=ts,
                text=summary_line,
                blocks=[{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": summary_line},
                }],
            )
        except Exception:
            logger.debug("Failed to update recommendation denial message", exc_info=True)

        await emit("recommendation_denied", {
            "approval_id": approval_id, "denier": denier_name,
        })

    # --- Recommendation refinement actions ---

    @app.action("refine_recommendation")
    async def handle_refine_recommendation(ack, body) -> None:
        await ack()
        approval_id = int(body["actions"][0]["value"])
        refiner_id = body["user"]["id"]
        refiner_name = await _resolve_user_name(refiner_id, settings)

        approval = await db.get_pending_rec_approval(approval_id)
        if not approval or approval["status"] != "pending":
            return

        await db.update_pending_rec_approval(approval_id, status="refining")

        client = AsyncWebClient(token=settings.slack_bot_token)
        ch = approval.get("approval_channel") or body["channel"]["id"]
        ts = approval.get("approval_message_ts") or body["message"]["ts"]

        # Replace the buttons with a "Refinement in progress..." message
        try:
            rec_ids = json.loads(approval.get("recommendation_ids", "[]"))
            rec_lines_parts: list[str] = []
            for rid in rec_ids:
                rec = await db.get_recommendation(rid)
                if rec:
                    rec_lines_parts.append(
                        f"• `{rec['canonical_form']}` ({rec.get('category', 'general')})"
                    )
            rec_lines = "\n".join(rec_lines_parts)
            preview = approval["original_text"][:500]
            if len(approval["original_text"]) > 500:
                preview += "..."

            status_text = (
                f":mag: *Recommendation approval needed*\n\n"
                f"*User:* {approval.get('user_name', 'unknown')}\n\n"
                f"*Recommendations requiring approval:*\n{rec_lines}\n\n"
                f"*Response preview:*\n>>>{preview}\n\n"
                f":pencil2: _Refinement in progress by {refiner_name}..._"
            )
            await client.chat_update(
                channel=ch,
                ts=ts,
                text=status_text,
                blocks=[{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": status_text},
                }],
            )
        except Exception:
            logger.debug("Failed to update approval message for refinement", exc_info=True)

        # Post thread reply with refinement context
        try:
            rec_ids = json.loads(approval.get("recommendation_ids", "[]"))
            recommendations: list[dict] = []
            for rid in rec_ids:
                rec = await db.get_recommendation(rid)
                if rec:
                    recommendations.append(rec)

            # Try to get user issue summary from conversation history
            conv_key = (approval["channel_id"], approval.get("thread_ts", ""))
            conv = _conversations.get(conv_key, [])
            issue_summary = ""
            for msg in conv:
                if msg["role"] == "user":
                    issue_summary = msg["content"][:300]
                    break

            context_blocks = format_refinement_context_blocks(
                original_text=approval["original_text"],
                recommendations=recommendations,
                issue_summary=issue_summary,
            )
            await client.chat_postMessage(
                channel=ch,
                thread_ts=ts,
                text="Refinement context",
                blocks=context_blocks,
            )
        except Exception:
            logger.debug("Failed to post refinement context thread", exc_info=True)

        # Register the refinement thread and seed conversation context
        _active_refinements[(ch, ts)] = approval_id
        refinement_key = (ch, ts)
        _conversations[refinement_key] = [
            {
                "role": "assistant",
                "content": (
                    f"Here is the original response that needs refinement:\n\n"
                    f"{approval['original_text']}\n\n"
                    f"I'm ready to help you refine this response. "
                    f"What changes would you like to make?"
                ),
            },
        ]

        await emit("recommendation_refine_started", {
            "approval_id": approval_id, "refiner": refiner_name,
        })

    async def _handle_refinement_message(
        event: dict, text: str, settings: Settings,
    ) -> None:
        """Handle a message in a refinement thread — run the agent with REFINEMENT_SYSTEM_PROMPT."""
        channel = event.get("channel", "")
        thread_ts = event.get("thread_ts", "")
        key = (channel, thread_ts)

        approval_id = _active_refinements.get(key)
        if approval_id is None:
            return

        # Append engineer's message to conversation
        conv = _conversations.setdefault(key, [])
        conv.append({"role": "user", "content": text})

        # Run the agent with the refinement system prompt (no tools)
        agent = _get_agent(settings)
        try:
            result = await agent.run(
                messages=conv,
                system_prompt=REFINEMENT_SYSTEM_PROMPT,
            )
        except Exception:
            logger.warning("Refinement agent call failed", exc_info=True)
            return

        # Store the refined text
        conv.append({"role": "assistant", "content": result.text})
        await db.update_pending_rec_approval(approval_id, refined_text=result.text)

        # Post the response + Send/Continue buttons in thread
        client = AsyncWebClient(token=settings.slack_bot_token)
        try:
            response_blocks = format_response_blocks(result.text, settings.sn_instance_url)
            send_blocks = format_refinement_send_button(approval_id)
            await client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=result.text,
                blocks=response_blocks + send_blocks,
            )
        except Exception:
            logger.debug("Failed to post refinement response", exc_info=True)

    @app.action("send_refined_recommendation")
    async def handle_send_refined_recommendation(ack, body) -> None:
        await ack()
        approval_id = int(body["actions"][0]["value"])
        sender_id = body["user"]["id"]
        sender_name = await _resolve_user_name(sender_id, settings)

        approval = await db.get_pending_rec_approval(approval_id)
        if not approval or approval["status"] != "refining":
            return

        refined_text = approval.get("refined_text")
        if not refined_text:
            return

        await db.update_pending_rec_approval(approval_id, status="approved")

        # Increment approval counts (refined = acceptable)
        rec_ids = json.loads(approval.get("recommendation_ids", "[]"))
        for rec_id in rec_ids:
            updated_rec = await db.increment_recommendation_approval(rec_id)
            if (
                updated_rec
                and updated_rec["status"] != "trusted"
                and updated_rec["approval_count"] >= settings.recommendation_trust_threshold
            ):
                await db.update_recommendation(rec_id, status="trusted")

        # Check if conversation moved to an incident channel
        orig_channel = approval["channel_id"]
        orig_thread = approval.get("thread_ts", "")
        incident_ch = _find_incident_channel_for_thread(orig_channel, orig_thread)

        # Deliver the refined response
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            sn_url = settings.sn_instance_url
            linked = linkify_servicenow_refs(refined_text, sn_url)
            blocks = format_response_blocks(refined_text, sn_url)

            if incident_ch:
                # Post as a new message in the incident channel
                await client.chat_postMessage(
                    channel=incident_ch,
                    text=linked,
                    blocks=blocks,
                )
                # Update live summary with the refined recommendation
                await _update_incident_summary(incident_ch, refined_text, settings)
                inc_ctx = _incident_context.get(incident_ch, {})
                inc_tid = inc_ctx.get("ticket_id")
                if inc_tid:
                    _push_live_summary_now(inc_tid, settings)
            else:
                # Update the redacted message in the original thread
                await client.chat_update(
                    channel=orig_channel,
                    ts=approval["message_ts"],
                    text=linked,
                    blocks=blocks,
                )
                # Post thread notification to the user
                if orig_thread:
                    await client.chat_postMessage(
                        channel=orig_channel,
                        thread_ts=orig_thread,
                        text=(
                            ":pencil2: The troubleshooting steps for this issue "
                            "have been reviewed, refined, and approved by IT. "
                            "Please see the updated message above."
                        ),
                    )
        except Exception:
            logger.debug("Failed to update user message with refined recommendation", exc_info=True)

        # Update the #it-helpdesk approval message with summary
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            ch = approval.get("approval_channel") or body["channel"]["id"]
            ts = approval.get("approval_message_ts") or body["message"]["ts"]
            summary_line = await _build_rec_decision_line(
                ":pencil2:", "refined and sent", sender_name,
                rec_ids, approval, settings,
            )
            await client.chat_update(
                channel=ch,
                ts=ts,
                text=summary_line,
                blocks=[{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": summary_line},
                }],
            )
        except Exception:
            logger.debug("Failed to update refined recommendation approval message", exc_info=True)

        # Clean up tracking state
        for key, aid in list(_active_refinements.items()):
            if aid == approval_id:
                _active_refinements.pop(key, None)
                _conversations.pop(key, None)
                break

        # Dismiss Send/Continue buttons and show confirmation with nav link
        link_text = ""
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            conv_channel = approval.get("channel_id", "")
            conv_thread = approval.get("thread_ts", "")
            if conv_thread and conv_channel:
                link_resp = await client.chat_getPermalink(
                    channel=conv_channel, message_ts=conv_thread,
                )
                permalink = link_resp.get("permalink", "")
                if permalink:
                    link_text = f" <{permalink}|View conversation>"
            elif conv_channel:
                link_text = f" <#{conv_channel}>"
        except Exception:
            logger.debug("Failed to build nav link for refined recommendation", exc_info=True)

        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            msg = body.get("message", {})
            original_blocks = msg.get("blocks", [])
            text_blocks = [b for b in original_blocks if b.get("type") != "actions"]
            text_blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":white_check_mark: Refined response sent to the user.{link_text}",
                },
            })
            await client.chat_update(
                channel=body["channel"]["id"],
                ts=msg["ts"],
                text=msg.get("text", ""),
                blocks=text_blocks,
            )
        except Exception:
            logger.debug("Failed to dismiss send refined buttons", exc_info=True)

        await emit("recommendation_refined", {
            "approval_id": approval_id, "sender": sender_name,
        })

    @app.action("continue_refining")
    async def handle_continue_refining(ack, body) -> None:
        """Legacy handler — button removed, but ack any stale clicks."""
        await ack()

    # --- Collaborative review actions ---

    async def _build_collab_decision_line(
        emoji: str, action: str, actor_name: str,
        review: dict, settings: Settings,
    ) -> str:
        """Build a single-line summary for #it-helpdesk after collab approve/modify/takeover."""
        summary_text = review.get("issue_summary", "") or review.get("ticket_id", "")
        if len(summary_text) > 120:
            summary_text = summary_text[:117] + "..."

        link_text = ""
        conv_channel = review.get("channel_id", "")
        conv_thread = review.get("thread_ts", "")
        if conv_channel:
            try:
                client = AsyncWebClient(token=settings.slack_bot_token)
                msg_ts = conv_thread or review.get("placeholder_message_ts", "")
                if msg_ts:
                    link_resp = await client.chat_getPermalink(
                        channel=conv_channel, message_ts=msg_ts,
                    )
                    permalink = link_resp.get("permalink", "")
                    if permalink:
                        link_text = f" | <{permalink}|View thread>"
            except Exception:
                logger.debug("Failed to get permalink for collab decision line", exc_info=True)

        return (
            f"{emoji} Collaborative review {action} by {actor_name}"
            f" — _{summary_text}_{link_text}"
        )

    @app.action("collab_approve")
    async def handle_collab_approve(ack, body) -> None:
        await ack()
        review_id = int(body["actions"][0]["value"])
        approver_id = body["user"]["id"]
        approver_name = await _resolve_user_name(approver_id, settings)

        review = await db.get_collaborative_review(review_id)
        if not review or review["status"] != "pending":
            return

        await db.update_collaborative_review(
            review_id, status="approved", resolved_by=approver_name,
        )

        # Update user's placeholder message with the full original response
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            sn_url = settings.sn_instance_url
            original = review["original_response"]
            blocks = format_response_blocks(original, sn_url)
            linked = linkify_servicenow_refs(original, sn_url)
            await client.chat_update(
                channel=review["channel_id"],
                ts=review["placeholder_message_ts"],
                text=linked,
                blocks=blocks,
            )
            # Post thread notification so user gets a Slack ping
            thread = review.get("thread_ts", "")
            if thread:
                await client.chat_postMessage(
                    channel=review["channel_id"],
                    thread_ts=thread,
                    text=(
                        ":white_check_mark: The response for this issue has been "
                        "reviewed and approved by IT. Please see the updated message above."
                    ),
                )
        except Exception:
            logger.debug("Failed to update user message on collab approve", exc_info=True)

        # Update the #it-helpdesk message with summary + link
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            ch = body["channel"]["id"]
            ts = body["message"]["ts"]
            summary_line = await _build_collab_decision_line(
                ":white_check_mark:", "approved", approver_name, review, settings,
            )
            await client.chat_update(
                channel=ch, ts=ts, text=summary_line,
                blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": summary_line}}],
            )
        except Exception:
            logger.debug("Failed to update collab approval message", exc_info=True)

        await emit("collab_review_approved", {
            "review_id": review_id, "approver": approver_name,
        })

    @app.action("collab_modify")
    async def handle_collab_modify(ack, body) -> None:
        await ack()
        review_id = int(body["actions"][0]["value"])
        trigger_id = body["trigger_id"]

        review = await db.get_collaborative_review(review_id)
        if not review or review["status"] != "pending":
            return

        # Open a Slack modal pre-filled with the original response
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            await client.views_open(
                trigger_id=trigger_id,
                view={
                    "type": "modal",
                    "callback_id": "collab_modify_submit",
                    "private_metadata": str(review_id),
                    "title": {"type": "plain_text", "text": "Modify Response"},
                    "submit": {"type": "plain_text", "text": "Send to User"},
                    "close": {"type": "plain_text", "text": "Cancel"},
                    "blocks": [
                        {
                            "type": "input",
                            "block_id": "response_block",
                            "label": {"type": "plain_text", "text": "Modified Response"},
                            "element": {
                                "type": "plain_text_input",
                                "action_id": "response_input",
                                "multiline": True,
                                "initial_value": review["original_response"][:3000],
                            },
                        }
                    ],
                },
            )
        except Exception:
            logger.warning("Failed to open collab modify modal", exc_info=True)

    @app.view("collab_modify_submit")
    async def handle_collab_modify_submit(ack, body) -> None:
        await ack()
        review_id = int(body["view"]["private_metadata"])
        modifier_id = body["user"]["id"]
        modifier_name = await _resolve_user_name(modifier_id, settings)

        modified_text = (
            body["view"]["state"]["values"]
            ["response_block"]["response_input"]["value"]
        )

        review = await db.get_collaborative_review(review_id)
        if not review or review["status"] != "pending":
            return

        await db.update_collaborative_review(
            review_id,
            status="modified",
            resolved_by=modifier_name,
            modified_response=modified_text,
        )

        # Update user's placeholder with the modified response
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            sn_url = settings.sn_instance_url
            blocks = format_response_blocks(modified_text, sn_url)
            linked = linkify_servicenow_refs(modified_text, sn_url)
            await client.chat_update(
                channel=review["channel_id"],
                ts=review["placeholder_message_ts"],
                text=linked,
                blocks=blocks,
            )
            thread = review.get("thread_ts", "")
            if thread:
                await client.chat_postMessage(
                    channel=review["channel_id"],
                    thread_ts=thread,
                    text=(
                        ":white_check_mark: The response for this issue has been "
                        "reviewed and updated by IT. Please see the updated message above."
                    ),
                )
        except Exception:
            logger.debug("Failed to update user message on collab modify", exc_info=True)

        # Update #it-helpdesk message with summary + link
        helpdesk_ts = review.get("helpdesk_message_ts")
        if helpdesk_ts and settings.it_helpdesk_channel_id:
            try:
                client = AsyncWebClient(token=settings.slack_bot_token)
                summary_line = await _build_collab_decision_line(
                    ":pencil2:", "modified and sent", modifier_name, review, settings,
                )
                await client.chat_update(
                    channel=settings.it_helpdesk_channel_id,
                    ts=helpdesk_ts, text=summary_line,
                    blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": summary_line}}],
                )
            except Exception:
                logger.debug("Failed to update collab modify helpdesk message", exc_info=True)

        await emit("collab_review_modified", {
            "review_id": review_id, "modifier": modifier_name,
        })

    @app.action("collab_takeover")
    async def handle_collab_takeover(ack, body) -> None:
        await ack()
        review_id = int(body["actions"][0]["value"])
        takeover_id = body["user"]["id"]
        takeover_name = await _resolve_user_name(takeover_id, settings)

        review = await db.get_collaborative_review(review_id)
        if not review or review["status"] != "pending":
            return

        await db.update_collaborative_review(
            review_id, status="taken_over", resolved_by=takeover_name,
        )

        # Update user's placeholder to indicate takeover
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            takeover_text = (
                ":bust_in_silhouette: _A Support Agent is taking over this issue. "
                "They'll be with you shortly._"
            )
            await client.chat_update(
                channel=review["channel_id"],
                ts=review["placeholder_message_ts"],
                text=takeover_text,
                blocks=[{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": takeover_text},
                }],
            )
            # Post thread notification
            thread = review.get("thread_ts", "")
            if thread:
                await client.chat_postMessage(
                    channel=review["channel_id"],
                    thread_ts=thread,
                    text=(
                        f":bust_in_silhouette: {takeover_name} from IT is taking "
                        f"over this issue and will assist you directly."
                    ),
                )

            # If an incident channel exists for this ticket, invite IT staff
            ticket_id = review.get("ticket_id", "")
            if ticket_id:
                for ch_id, ctx in _incident_context.items():
                    if ctx.get("ticket_id") == ticket_id:
                        try:
                            await client.conversations_invite(
                                channel=ch_id, users=takeover_id,
                            )
                        except Exception:
                            pass  # already_in_channel or other
                        break
        except Exception:
            logger.debug("Failed to update user message on collab takeover", exc_info=True)

        # Update the #it-helpdesk message with summary + link
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            ch = body["channel"]["id"]
            ts = body["message"]["ts"]
            summary_line = await _build_collab_decision_line(
                ":bust_in_silhouette:", "taken over", takeover_name, review, settings,
            )
            await client.chat_update(
                channel=ch, ts=ts, text=summary_line,
                blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": summary_line}}],
            )
        except Exception:
            logger.debug("Failed to update collab takeover message", exc_info=True)

        await emit("collab_review_takeover", {
            "review_id": review_id, "taken_over_by": takeover_name,
        })

    # --- KB suggestion actions (collaborative review) ---

    @app.action("collab_create_kb")
    async def handle_collab_create_kb(ack, body) -> None:
        await ack()
        review_id = int(body["actions"][0]["value"])
        creator_id = body["user"]["id"]
        creator_name = await _resolve_user_name(creator_id, settings)

        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            await client.chat_update(
                channel=body["channel"]["id"],
                ts=body["message"]["ts"],
                text=f":books: KB article creation accepted by {creator_name}. Please create in ServiceNow.",
                blocks=[{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":books: KB article creation accepted by {creator_name}. Please create in ServiceNow.",
                    },
                }],
            )
        except Exception:
            logger.debug("Failed to update KB creation message", exc_info=True)

        await emit("collab_kb_accepted", {
            "review_id": review_id, "creator": creator_name,
        })

    @app.action("collab_dismiss_kb")
    async def handle_collab_dismiss_kb(ack, body) -> None:
        await ack()
        dismisser_id = body["user"]["id"]
        dismisser_name = await _resolve_user_name(dismisser_id, settings)

        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            await client.chat_update(
                channel=body["channel"]["id"],
                ts=body["message"]["ts"],
                text=f":no_entry: KB suggestion dismissed by {dismisser_name}.",
                blocks=[{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":no_entry: KB suggestion dismissed by {dismisser_name}.",
                    },
                }],
            )
        except Exception:
            logger.debug("Failed to update KB dismissal message", exc_info=True)

    # --- Move to private channel action ---

    @app.action("move_to_private_channel")
    async def handle_move_to_private_channel(ack, body) -> None:
        await ack()
        raw_value = body["actions"][0]["value"]
        channel = body["channel"]["id"]
        message_ts = body["message"]["ts"]
        thread_ts = body["message"].get("thread_ts") or message_ts
        button_user_id = body["user"]["id"]

        client = AsyncWebClient(token=settings.slack_bot_token)

        # Parse ticket_id and requester from button value (format: "INC...:U...")
        if ":" in raw_value:
            ticket_id, requester_id = raw_value.split(":", 1)
        else:
            # Legacy buttons without requester encoded
            ticket_id = raw_value
            requester_id = None

        # Only the requester can use this button
        if requester_id and button_user_id != requester_id:
            try:
                clicker_info = await client.users_info(user=button_user_id)
                clicker_profile = clicker_info["user"]["profile"]
                first_name = (
                    clicker_profile.get("first_name")
                    or clicker_profile.get("display_name", "").split()[0]
                    or clicker_profile.get("real_name", "").split()[0]
                    or "there"
                )
            except Exception:
                first_name = "there"
            try:
                await client.chat_postEphemeral(
                    channel=channel,
                    user=button_user_id,
                    text=(
                        f"Hi {first_name}, this button is only for the person "
                        f"who reported the issue. If you need help, please "
                        f"start a new thread in this channel."
                    ),
                    thread_ts=thread_ts,
                )
            except Exception:
                logger.debug("Failed to send requester-only ephemeral", exc_info=True)
            return

        # Check if channel already exists (idempotent)
        for ch_id, ctx in _incident_context.items():
            if ctx.get("ticket_id") == ticket_id:
                # Ensure the clicker is in the channel
                try:
                    await client.conversations_invite(channel=ch_id, users=button_user_id)
                except Exception:
                    pass  # already_in_channel or other — not critical
                # Replace button with confirmation + link
                try:
                    await client.chat_update(
                        channel=channel,
                        ts=message_ts,
                        text=f":white_check_mark: *{ticket_id}* — Head over to <#{ch_id}> to continue.",
                        blocks=[{
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f":white_check_mark: *{ticket_id}* — Head over to <#{ch_id}> to continue.",
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

        # Replace the button with confirmation + link (single message, no extras)
        try:
            await client.chat_update(
                channel=channel,
                ts=message_ts,
                text=f":white_check_mark: *{ticket_id}* — Head over to <#{channel_id}> to continue.",
                blocks=[{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":white_check_mark: *{ticket_id}* — Head over to <#{channel_id}> to continue.",
                    },
                }],
            )
        except Exception:
            logger.debug("Failed to update button message", exc_info=True)

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
    # General
    "restart", "reboot", "check cables", "clear cache", "clear browser cache",
    "try again", "log out and log back in", "sign out and sign back in",
    "check internet connection", "check your internet", "update your browser",
    "close and reopen", "power cycle", "unplug and replug",
    # VPN / WiFi
    "connect vpn", "disconnect vpn", "reconnect vpn", "disconnect and reconnect",
    "connect to vpn", "ensure vpn is connected", "check vpn", "toggle vpn",
    "connect wifi", "reconnect wifi", "disconnect and reconnect wifi",
    "forget network and reconnect", "rejoin network",
    # Mac hardware troubleshooting
    "reset smc", "smc reset", "perform smc reset",
    "reset nvram", "nvram reset", "reset pram", "pram reset",
    "boot safe mode", "safe mode", "boot into safe mode",
    "hold power button", "press and hold power",
    "reset battery", "recalibrate battery",
    # Software updates
    "install updates", "install pending updates", "check for updates",
    "software update", "update macos", "update windows", "update os",
    "install macos update", "install windows update",
    # Windows hardware troubleshooting
    "run sfc", "system file checker", "check disk",
    "boot safe mode windows", "safe mode with networking",
    # Network safety / isolation advice
    "disconnect from network", "keep disconnected from network",
    "disconnect from wifi", "keep laptop disconnected",
    "disconnect ethernet", "unplug ethernet",
    "disable wifi", "turn off wifi",
    "enable airplane mode", "turn on airplane mode",
}


_HIGH_RISK_CATEGORIES = {"core_app_violation"}
_HIGH_RISK_KEYWORDS = {
    "uninstall", "remove application", "disable security", "disable antivirus",
    "disable firewall", "disable endpoint", "stop security service",
    "modify registry", "delete system", "factory reset", "wipe",
}


def _classify_risk_level(recommendations: list[dict]) -> str:
    """Returns 'high', 'medium', or 'low'."""
    for rec in recommendations:
        if rec.get("category") in _HIGH_RISK_CATEGORIES:
            return "high"
        canonical = rec.get("canonical_form", "").lower()
        if any(kw in canonical for kw in _HIGH_RISK_KEYWORDS):
            return "high"
    return "low"  # medium/low both use standard gating


# ---------------------------------------------------------------------------
# Processing indicator helper
# ---------------------------------------------------------------------------

async def _post_processing_indicator(say, **kwargs) -> str:
    """Post a processing indicator and return its message ts."""
    try:
        msg = await say(
            text=":hourglass_flowing_sand: _Processing your request..._",
            **kwargs,
        )
    except Exception:
        return ""
    if msg is None:
        return ""
    if isinstance(msg, dict):
        return msg.get("ts", "")
    if hasattr(msg, "data"):
        return msg.data.get("ts", "")
    return ""


# ---------------------------------------------------------------------------
# User profile helpers
# ---------------------------------------------------------------------------

def _build_profile_context(profile: dict) -> str:
    """Format a user profile dict into a readable string for the system prompt."""
    labels = {
        "device_type": "Device",
        "os": "OS",
        "technical_level": "Technical level",
        "role": "Role",
        "department": "Department",
        "notes": "Notes",
    }
    lines = []
    for key, label in labels.items():
        value = profile.get(key, "")
        if value:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


async def _extract_user_profile_updates(
    conversation_history: list[dict], user_id: str, settings: Settings,
) -> dict:
    """Use Gemini to extract user details revealed during the conversation."""
    try:
        formatted = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in conversation_history[-10:]
        )
        client = genai.Client(api_key=settings.gemini_api_key)
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=(
                "Analyze this IT support conversation and extract STABLE factual details "
                "about the user that the USER explicitly stated. Only include information "
                "the user directly revealed — do not guess, infer, or extract details from "
                "the assistant's responses or assumptions.\n\n"
                "IMPORTANT: Only extract PERMANENT user attributes — things that stay true "
                "across multiple support tickets. Do NOT extract:\n"
                "- The specific issue or symptoms being discussed (those are ticket-specific)\n"
                "- Software or applications mentioned in passing or as part of troubleshooting\n"
                "- Applications the assistant assumed or suggested\n"
                "- Temporary states (e.g., 'laptop is slow today')\n\n"
                "Return a JSON object with only the fields you found (omit fields with no information):\n"
                '- "device_type": specific device model the user said they use (e.g., "MacBook Pro", "Dell Latitude 5540")\n'
                '- "os": operating system the user said they run (e.g., "macOS Sonoma", "Windows 11")\n'
                '- "technical_level": "beginner", "intermediate", or "advanced" based on how they '
                "describe their issue and interact\n"
                '- "role": job title or role if the user mentioned it\n'
                '- "department": department if the user mentioned it\n'
                '- "notes": ONLY permanent details like hardware setup (e.g., "uses dual monitors", '
                '"has docking station") — NOT applications, symptoms, or issue-specific details\n\n'
                "Return {} if no new STABLE user information was revealed.\n\n"
                "IMPORTANT: Return ONLY the JSON object, no other text.\n\n"
                f"Conversation:\n{formatted}"
            ),
        )
        raw = (response.text or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        if not isinstance(result, dict):
            return {}
        # Only keep valid fields
        valid_keys = {"device_type", "os", "technical_level", "role", "department", "notes"}
        return {k: v for k, v in result.items() if k in valid_keys and v}
    except Exception:
        logger.debug("Failed to extract user profile updates", exc_info=True)
        return {}


async def _update_user_profile_from_conversation(
    history: list[dict], user_id: str, settings: Settings,
) -> None:
    """Extract profile updates from conversation and persist them (fire-and-forget)."""
    try:
        updates = await _extract_user_profile_updates(history, user_id, settings)
        if updates:
            await db.upsert_user_profile(user_id, **updates)
            logger.info("Updated user profile for %s: %s", user_id, list(updates.keys()))
    except Exception:
        logger.debug("Failed to update user profile from conversation", exc_info=True)


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
                "Include recommendations to disable, enable, or remove browser extensions "
                "(e.g., 'disable your Chrome extensions', 'try disabling extensions one by one'). "
                "These are NOT basic advice — they affect the user's browser configuration and must "
                "be extracted.\n\n"
                "Also include application-specific feature changes or built-in tool usage "
                "(e.g., 'enable Memory Saver in Chrome', 'use Chrome Task Manager to kill tabs', "
                "'enable efficiency mode in Edge', 'turn on hardware acceleration'). "
                "These are settings or feature toggles within specific applications.\n\n"
                "Do NOT include basic/generic advice like: restart, reboot, check cables, "
                "clear cache, try again, log out/in, check internet, update browser, "
                "close and reopen, power cycle.\n\n"
                "CRITICAL — Core Application Protection:\n"
                "These are IT-managed apps that CANNOT be uninstalled: GlobalProtect, "
                "CrowdStrike/Falcon, Microsoft Intune/Company Portal, Jamf, Cisco AnyConnect, "
                "Zscaler, Microsoft Defender, SentinelOne.\n"
                "If the response suggests uninstalling, removing, or disabling ANY of these, "
                'flag with category "core_app_violation". This must NEVER reach users without '
                "IT review.\n\n"
                "For each recommendation found, output a JSON array of objects with:\n"
                '- "original_text": the COMPLETE text block from the response for this '
                "recommendation. This must include the ENTIRE numbered step or bullet point "
                "AND all of its sub-bullets, sub-steps, and continuation lines — from the "
                "step number/bullet through to just before the next top-level step or section. "
                "Copy the text EXACTLY as it appears, preserving all formatting, newlines, "
                "and sub-items. Do NOT extract just a phrase or single line — extract the "
                "whole block.\n"
                '- "canonical_form": a short normalized version (e.g. "disable hardware acceleration in Chrome")\n'
                '- "category": one of "settings_change", "install", "command", "config_change", '
                '"feature_toggle", "core_app_violation"\n\n'
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
            canonical_words = set(canonical.split())
            is_basic = any(
                set(basic.split()).issubset(canonical_words)
                for basic in _BASIC_RECOMMENDATIONS
            )
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


async def _open_collaborative_review(
    response_text: str,
    channel: str,
    thread_ts: str | None,
    say,
    settings: Settings,
    *,
    recommendations: list[dict],
    interaction_id: int | None = None,
    user_name: str = "a user",
) -> tuple[str, str | None]:
    """Open a collaborative review thread in #it-helpdesk for high-risk responses.

    Posts a placeholder to the user and sends the full response to IT staff for review.
    Returns ``(placeholder_text, posted_ts)``.
    """
    # Build trigger reason from recommendations
    trigger_parts = []
    for rec in recommendations:
        canonical = rec.get("canonical_form", "")
        if canonical:
            trigger_parts.append(canonical)
    trigger_reason = "; ".join(trigger_parts) if trigger_parts else "high-risk recommendation"

    # Look up ticket_id from _ticket_threads
    ticket_id = ""
    if thread_ts:
        for tid, (ch, ts) in _ticket_threads.items():
            if ch == channel and ts == thread_ts:
                ticket_id = tid
                break

    # Resolve user_slack_id from conversation context
    user_slack_id = ""
    conv_key = (channel, thread_ts or "")
    for key, iid in _interaction_ids.items():
        if key == conv_key:
            interaction = await db.get_interaction(iid)
            if interaction:
                user_slack_id = interaction.get("requester_slack_id", "")
            break
    if not user_slack_id:
        # Fallback: try to get from channel context
        for tid, (ch, ts) in _ticket_threads.items():
            if ch == channel and ts == (thread_ts or ""):
                interaction = await db.get_interaction_by_ticket(tid)
                if interaction:
                    user_slack_id = interaction.get("requester_slack_id", "")
                break

    # Build issue summary from conversation history
    issue_summary = ""
    history = _conversations.get(conv_key, [])
    if history:
        user_msgs = [m["content"] for m in history if m["role"] == "user"]
        if user_msgs:
            first_msg = user_msgs[0]
            # Strip system context prefixes
            if first_msg.startswith("[System"):
                first_msg = first_msg.split("]\n\n", 1)[-1] if "]\n\n" in first_msg else first_msg
            issue_summary = first_msg[:300]

    # Post placeholder to user
    placeholder_text = (
        ":hourglass_flowing_sand: _I'm checking with the IT team on the best approach "
        "for this issue. You'll be updated shortly._"
    )
    say_kwargs: dict = {"text": placeholder_text}
    if thread_ts:
        say_kwargs["thread_ts"] = thread_ts

    msg_response = await say(**say_kwargs)
    posted_ts = ""
    if msg_response is not None:
        if isinstance(msg_response, dict):
            posted_ts = msg_response.get("ts", "")
        elif hasattr(msg_response, "get"):
            posted_ts = msg_response.get("ts", "")
        elif hasattr(msg_response, "data"):
            posted_ts = msg_response.data.get("ts", "")

    # Create DB record
    try:
        review_id = await db.create_collaborative_review(
            channel_id=channel,
            thread_ts=thread_ts or "",
            user_slack_id=user_slack_id,
            user_name=user_name,
            original_response=response_text,
            trigger_reason=trigger_reason,
            issue_summary=issue_summary,
            risk_level="high",
            interaction_id=interaction_id,
            ticket_id=ticket_id or None,
            placeholder_message_ts=posted_ts,
        )
    except Exception:
        logger.warning("Failed to create collaborative review record", exc_info=True)
        return placeholder_text, posted_ts

    # Post thread-starting message in #it-helpdesk
    if settings.it_helpdesk_channel_id:
        try:
            client = AsyncWebClient(token=settings.slack_bot_token)
            review_blocks = format_collaborative_review_blocks(
                review_id=review_id,
                trigger_reason=trigger_reason,
                original_response=response_text,
                user_name=user_name,
                ticket_id=ticket_id,
                channel_id=channel,
            )
            resp = await client.chat_postMessage(
                channel=settings.it_helpdesk_channel_id,
                text=f":rotating_light: Collaborative Review: {trigger_reason}",
                blocks=review_blocks,
            )
            helpdesk_ts = resp.get("ts", "")

            # Post thread reply with full bot response
            context_blocks = format_collaborative_context_blocks(
                original_response=response_text,
                issue_summary=issue_summary,
            )
            await client.chat_postMessage(
                channel=settings.it_helpdesk_channel_id,
                thread_ts=helpdesk_ts,
                text=f"Full bot response for review",
                blocks=context_blocks,
            )

            # Update DB with helpdesk message TS
            await db.update_collaborative_review(
                review_id,
                helpdesk_message_ts=helpdesk_ts,
            )
        except Exception:
            logger.warning("Failed to post collaborative review to #it-helpdesk", exc_info=True)

    return placeholder_text, posted_ts


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

    # 2.5. Risk check: high-risk → collaborative review
    risk_level = _classify_risk_level(recommendations)
    if risk_level == "high":
        return await _open_collaborative_review(
            response_text, channel, thread_ts, say, settings,
            recommendations=recommendations,
            interaction_id=interaction_id,
            user_name=user_name,
        )

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
    posted_ts = ""
    if msg_response is not None:
        if isinstance(msg_response, dict):
            posted_ts = msg_response.get("ts", "")
        elif hasattr(msg_response, "get"):
            posted_ts = msg_response.get("ts", "")
        elif hasattr(msg_response, "data"):
            posted_ts = msg_response.data.get("ts", "")

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

    # Post processing indicator immediately
    typing_ts = await _post_processing_indicator(say, thread_ts=thread_ts)

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

        # Fetch user profile for context
        profile = await db.get_user_profile(user_id)
        profile_context = _build_profile_context(profile) if profile else ""

        result: AgentResult = await agent.run(
            history,
            user_id=user_id,
            context_articles=list(_ai_context_articles.values()),
            user_profile_context=profile_context,
        )

        # Append assistant response to history
        history.append({"role": "assistant", "content": result.text})

        await _register_incident_channels(result)

        # Post debug reasoning trace (fire-and-forget)
        asyncio.create_task(_post_debug_summary(
            source="dm", user_id=user_id, user_message=text,
            result=result, channel=channel, thread_ts=thread_ts,
            settings=settings,
        ))

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
            if typing_ts:
                client = AsyncWebClient(token=settings.slack_bot_token)
                await client.chat_update(
                    channel=channel, ts=typing_ts,
                    text=linked_text, blocks=blocks,
                )
            else:
                await say(text=linked_text, blocks=blocks, thread_ts=thread_ts)
        else:
            # Recommendations were gated — delete the typing indicator
            if typing_ts:
                try:
                    client = AsyncWebClient(token=settings.slack_bot_token)
                    await client.chat_delete(channel=channel, ts=typing_ts)
                except Exception:
                    pass

        # Post public article feedback buttons + approval requests
        await _post_public_article_followups(result, channel, thread_ts, settings)

        # Update user profile from conversation (fire-and-forget)
        asyncio.create_task(
            _update_user_profile_from_conversation(history, user_id, settings)
        )

    except Exception as exc:
        logger.exception("Error processing message")
        await emit("error", {"source": "dm", "message": str(exc)[:300]})
        # Delete the typing indicator on error
        if typing_ts:
            try:
                client = AsyncWebClient(token=settings.slack_bot_token)
                await client.chat_delete(channel=channel, ts=typing_ts)
            except Exception:
                pass
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
        try:
            await asyncio.sleep(_SUMMARY_DEBOUNCE_SECONDS)
            await _push_live_summary(ticket_id, settings)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Live summary push failed for %s", ticket_id)
        finally:
            _summary_timers.pop(ticket_id, None)

    _summary_timers[ticket_id] = asyncio.create_task(_delayed())


def _push_live_summary_now(ticket_id: str, settings: Settings) -> None:
    """Push live summary immediately (fire-and-forget). Also resets the
    debounce timer so we don't double-push shortly after."""
    # Cancel any pending debounced push
    existing = _summary_timers.pop(ticket_id, None)
    if existing and not existing.done():
        existing.cancel()

    async def _immediate():
        try:
            await _push_live_summary(ticket_id, settings)
        except Exception:
            logger.exception("Immediate live summary push failed for %s", ticket_id)

    asyncio.create_task(_immediate())


async def _push_live_summary(ticket_id: str, settings: Settings) -> None:
    """Generate an AI summary of all conversations for this ticket and
    append it to the ServiceNow description field (below the original text,
    separated by 3 blank lines).  Each push replaces the previous summary."""
    logger.info("Starting live summary push for %s", ticket_id)

    # 1. Collect conversation history from both thread and channel
    history = _collect_ticket_history(ticket_id)
    if not history:
        logger.info("No conversation history found for %s, skipping summary", ticket_id)
        return

    # 2. Generate summary via Gemini
    summary = await _summarize_conversation(history, settings)

    # 3. Build updated description: original + 3 blank lines + summary
    _SUMMARY_SEPARATOR = "\n\n\n\n--- LIVE SUMMARY (auto-updated) ---\n"

    original = _ticket_original_descriptions.get(ticket_id, "")
    sn = ServiceNowClient(settings.sn_instance_url, settings.sn_username, settings.sn_password)
    try:
        incident = await sn.get_incident(ticket_id)
        if not incident:
            logger.warning("Incident %s not found in SN, skipping summary push", ticket_id)
            return

        # If we don't have the original cached, recover it from SN
        if not original:
            desc = incident.get("description", "")
            # Strip any previous live summary to get clean original text
            original = desc.split("--- LIVE SUMMARY")[0].rstrip()
            _ticket_original_descriptions[ticket_id] = original

        new_description = f"{original}{_SUMMARY_SEPARATOR}{summary}"

        await sn.update_incident(
            incident["sys_id"],
            {"description": new_description},
            current_state=incident.get("_raw_state", "1"),
        )
        logger.info("Pushed live summary to SN description for %s", ticket_id)
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

    # Post processing indicator immediately (no thread_ts for incident channels)
    typing_ts = await _post_processing_indicator(say)

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

        # Fetch user profile for context
        profile = await db.get_user_profile(user_id)
        profile_context = _build_profile_context(profile) if profile else ""

        result: AgentResult = await agent.run(
            history,
            user_id=user_id,
            context_articles=list(_ai_context_articles.values()),
            user_profile_context=profile_context,
        )

        history.append({"role": "assistant", "content": result.text})

        await _register_incident_channels(result)

        # Post debug reasoning trace (fire-and-forget)
        # Look up the original help-it thread for this incident channel
        inc_ctx = _incident_context.get(channel, {})
        inc_ticket_id = inc_ctx.get("ticket_id", "")
        help_thread_ts = ""
        if inc_ticket_id:
            ht = _ticket_threads.get(inc_ticket_id)
            if ht:
                help_thread_ts = ht[1]
        asyncio.create_task(_post_debug_summary(
            source="incident", user_id=user_id, user_message=text,
            result=result, channel=channel, thread_ts=help_thread_ts or None,
            settings=settings, incident_channel_id=channel,
            ticket_id=inc_ticket_id,
        ))

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
            if typing_ts:
                client = AsyncWebClient(token=settings.slack_bot_token)
                await client.chat_update(
                    channel=channel, ts=typing_ts,
                    text=linked_text, blocks=blocks,
                )
            else:
                await say(text=linked_text, blocks=blocks)
        else:
            # Recommendations were gated — delete the typing indicator
            if typing_ts:
                try:
                    client = AsyncWebClient(token=settings.slack_bot_token)
                    await client.chat_delete(channel=channel, ts=typing_ts)
                except Exception:
                    pass

        # Update the pinned summary message with progress
        await _update_incident_summary(channel, result.text, settings)

        # Push live summary to ServiceNow immediately after each agent response
        ctx = _incident_context.get(channel, {})
        inc_ticket_id = ctx.get("ticket_id")
        if inc_ticket_id:
            _push_live_summary_now(inc_ticket_id, settings)

        # Post public article feedback buttons + approval requests
        await _post_public_article_followups(result, channel, None, settings)

        # Update user profile from conversation (fire-and-forget)
        asyncio.create_task(
            _update_user_profile_from_conversation(history, user_id, settings)
        )

        # Notify #it-helpdesk when escalation happens in an incident channel
        if settings.it_helpdesk_channel_id:
            for tc in result.tool_calls:
                if tc["name"] != "update_ticket":
                    continue
                args = tc.get("args", {})
                if args.get("priority") not in ("high", "urgent"):
                    continue
                tc_ticket_id = args.get("ticket_id", "")
                if not tc_ticket_id:
                    continue
                try:
                    escalation_client = AsyncWebClient(token=settings.slack_bot_token)
                    user_display = await _resolve_user_name(user_id, settings)
                    esc_resp = await escalation_client.chat_postMessage(
                        channel=settings.it_helpdesk_channel_id,
                        text=(
                            f":rotating_light: *Escalation: {tc_ticket_id}*\n"
                            f"User <@{user_id}> ({user_display}) requested escalation "
                            f"in <#{channel}>.\n"
                            f"Priority set to *{args.get('priority')}*. "
                            f"Please join the channel to assist."
                        ),
                    )
                    # Store the escalation message ts for later resolution update
                    ctx = _incident_context.get(channel)
                    if ctx:
                        ctx["escalation_msg_ts"] = esc_resp["ts"]
                    logger.info("Posted escalation notice to #it-helpdesk for %s", tc_ticket_id)
                except Exception:
                    logger.debug("Failed to post escalation notice to #it-helpdesk", exc_info=True)

    except Exception as exc:
        logger.exception("Error processing incident channel message")
        await emit("error", {"source": "incident", "message": str(exc)[:300]})
        # Delete the typing indicator on error
        if typing_ts:
            try:
                client = AsyncWebClient(token=settings.slack_bot_token)
                await client.chat_delete(channel=channel, ts=typing_ts)
            except Exception:
                pass
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

    # Post processing indicator immediately
    typing_ts = await _post_processing_indicator(say, thread_ts=thread_ts)

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

        # Fetch user profile for context
        profile = await db.get_user_profile(user_id)
        profile_context = _build_profile_context(profile) if profile else ""

        result: AgentResult = await agent.run(
            history,
            user_id=user_id,
            context_articles=list(_ai_context_articles.values()),
            user_profile_context=profile_context,
        )

        history.append({"role": "assistant", "content": result.text})

        await _register_incident_channels(result)

        # Post debug reasoning trace (fire-and-forget)
        # Reverse-look up ticket + incident channel for this help-it thread
        help_ticket_id_dbg = ""
        help_inc_channel_dbg = ""
        for tid, (ch, ts) in _ticket_threads.items():
            if ch == channel and ts == thread_ts:
                help_ticket_id_dbg = tid
                break
        if help_ticket_id_dbg:
            for ch_id, ctx in _incident_context.items():
                if ctx.get("ticket_id") == help_ticket_id_dbg:
                    help_inc_channel_dbg = ch_id
                    break
        asyncio.create_task(_post_debug_summary(
            source="help-it", user_id=user_id, user_message=text,
            result=result, channel=channel, thread_ts=thread_ts,
            settings=settings, incident_channel_id=help_inc_channel_dbg,
            ticket_id=help_ticket_id_dbg,
        ))

        # Record interaction data
        iid = _interaction_ids.get(conv_key)
        if iid is not None:
            await _record_agent_result(iid, text, result)

        # Recommendation approval gate
        sn_url = settings.sn_instance_url
        user_display = await _resolve_user_name(user_id, settings)
        final_text, gated_msg_ts = await _gate_recommendations(
            result.text, channel, thread_ts, say, settings,
            interaction_id=iid, user_name=user_display,
        )
        if gated_msg_ts is None:
            linked_text = linkify_servicenow_refs(result.text, sn_url)
            blocks = format_response_blocks(result.text, sn_url)
            if typing_ts:
                client = AsyncWebClient(token=settings.slack_bot_token)
                await client.chat_update(
                    channel=channel, ts=typing_ts,
                    text=linked_text, blocks=blocks,
                )
            else:
                await say(text=linked_text, blocks=blocks, thread_ts=thread_ts)
        else:
            # Recommendations were gated — delete the typing indicator
            if typing_ts:
                try:
                    client = AsyncWebClient(token=settings.slack_bot_token)
                    await client.chat_delete(channel=channel, ts=typing_ts)
                except Exception:
                    pass

        # Update user profile from conversation (fire-and-forget)
        asyncio.create_task(
            _update_user_profile_from_conversation(history, user_id, settings)
        )

        # ── Post ticket follow-up with "Move to private channel" button ──
        if auto_ticket_info and auto_ticket_info.get("success"):
            ticket = auto_ticket_info.get("ticket", {})
            ticket_id = ticket.get("ticket_id", "")

            if ticket_id:
                # Skip button if a channel was already created (e.g. auto-escalation)
                already_has_channel = any(
                    ctx.get("ticket_id") == ticket_id for ctx in _incident_context.values()
                )
                if already_has_channel:
                    # Just post the ticket confirmation without the button
                    existing_ch = next(
                        (ch_id for ch_id, ctx in _incident_context.items()
                         if ctx.get("ticket_id") == ticket_id), None
                    )
                    followup_text = (
                        f":ticket: *Ticket {ticket_id} created.* "
                        f"A private channel <#{existing_ch}> has been created for this issue."
                    )
                    followup_text = linkify_servicenow_refs(followup_text, sn_url)
                    await say(
                        text=followup_text,
                        blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": followup_text}}],
                        thread_ts=thread_ts,
                    )
                else:
                    # Encode requester user_id in button value so only they can click
                    button_value = f"{ticket_id}:{user_id}"
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
                                    "value": button_value,
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

        # Push live summary to ServiceNow immediately after each agent response
        help_ticket_id: str | None = None
        for tid, (ch, ts) in _ticket_threads.items():
            if ch == channel and ts == thread_ts:
                help_ticket_id = tid
                break
        if help_ticket_id:
            _push_live_summary_now(help_ticket_id, settings)

    except Exception as exc:
        logger.exception("Error processing #help-it message")
        await emit("error", {"source": "help-it", "message": str(exc)[:300]})
        # Delete the typing indicator on error
        if typing_ts:
            try:
                client = AsyncWebClient(token=settings.slack_bot_token)
                await client.chat_delete(channel=channel, ts=typing_ts)
            except Exception:
                pass
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

    # Suggest KB article if this ticket had a collaborative review
    asyncio.create_task(_suggest_kb_article_for_collab(ticket_id, ticket_data, settings))

    # Post resolution notice with "Archive Now" button in the incident channel
    channel_id: str | None = None
    for ch_id, ctx in _incident_context.items():
        if ctx.get("ticket_id") == ticket_id:
            channel_id = ch_id
            break

    if channel_id is not None:
        # Update the escalation message in #it-helpdesk to show resolved status
        ctx = _incident_context.get(channel_id, {})
        escalation_ts = ctx.get("escalation_msg_ts")
        if escalation_ts and settings.it_helpdesk_channel_id:
            close_notes = ticket_data.get("close_notes", "")
            updated_text = f":white_check_mark: ~*Escalation: {ticket_id}*~ — *Resolved*"
            if close_notes:
                updated_text += f"\n>_{close_notes}_"
            updated_text = linkify_servicenow_refs(updated_text, settings.sn_instance_url)
            try:
                await client.chat_update(
                    channel=settings.it_helpdesk_channel_id,
                    ts=escalation_ts,
                    text=updated_text,
                )
            except Exception:
                logger.debug("Failed to update escalation message for %s", ticket_id, exc_info=True)

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

        # Escalate the ticket to high priority
        interaction_id = approval.get("interaction_id")
        if interaction_id:
            try:
                interaction = await db.get_interaction(interaction_id)
                if interaction and interaction.get("ticket_id"):
                    await update_ticket(
                        interaction["ticket_id"],
                        priority="high",
                        comment=f"Recommendation approval timed out after {_APPROVAL_TIMEOUT_MINUTES}min — escalating for agent follow-up.",
                        _settings=settings,
                    )
                    logger.info("Escalated ticket %s after recommendation timeout", interaction["ticket_id"])
            except Exception:
                logger.debug("Failed to escalate ticket on recommendation timeout", exc_info=True)

        # Update the user's message with escalation notice
        try:
            timeout_text = (
                f"{approval['redacted_text'].rsplit(':hourglass_flowing_sand:', 1)[0].rstrip()}"
                "\n\n:rotating_light: _The troubleshooting steps require further review. "
                "This issue has been escalated and a Support Agent will follow up with you._"
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

        # Update the #it-helpdesk message with summary + link
        msg_ts = approval.get("approval_message_ts")
        ch = approval.get("approval_channel")
        if msg_ts and ch:
            try:
                summary_line = await _build_rec_decision_line(
                    ":hourglass:", f"auto-denied (timed out after {_APPROVAL_TIMEOUT_MINUTES}min)",
                    "system", rec_ids, approval, settings,
                )
                await client.chat_update(
                    channel=ch,
                    ts=msg_ts,
                    text=summary_line,
                    blocks=[{
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": summary_line},
                    }],
                )
            except Exception:
                logger.debug(
                    "Failed to update expired recommendation approval message %d",
                    approval_id, exc_info=True,
                )

        await emit("recommendation_auto_denied", {"approval_id": approval_id})


# ---------------------------------------------------------------------------
# Collaborative review timeout loop
# ---------------------------------------------------------------------------

async def start_collaborative_review_timeout_loop(settings: Settings) -> None:
    """Background loop that handles timed-out collaborative reviews."""
    while True:
        await asyncio.sleep(_APPROVAL_CHECK_INTERVAL)
        try:
            await _process_expired_collaborative_reviews(settings)
        except Exception:
            logger.warning("Collaborative review timeout loop iteration failed", exc_info=True)


async def _process_expired_collaborative_reviews(settings: Settings) -> None:
    """Fall back to standard redact-and-approve for timed-out collaborative reviews."""
    timeout = settings.collaborative_review_timeout_minutes
    expired = await db.get_expired_collaborative_reviews(older_than_minutes=timeout)
    if not expired:
        return

    logger.info("Processing %d expired collaborative review(s)", len(expired))
    client = AsyncWebClient(token=settings.slack_bot_token)

    for review in expired:
        review_id = review["id"]
        await db.update_collaborative_review(review_id, status="timed_out")

        original = review["original_response"]
        channel_id = review["channel_id"]
        thread_ts = review.get("thread_ts", "")
        placeholder_ts = review.get("placeholder_message_ts", "")

        # Fall back: extract recommendations and use standard redact-and-approve
        recommendations = await _extract_recommendations(original, settings)
        if recommendations:
            recommendations = await _match_recommendations_to_db(recommendations, settings)
            untrusted = [r for r in recommendations if not r.get("is_trusted")]
            if untrusted:
                redacted_text = redact_recommendations(original, untrusted)

                # Update user's placeholder with redacted text
                if placeholder_ts:
                    try:
                        sn_url = settings.sn_instance_url
                        blocks = format_response_blocks(redacted_text, sn_url)
                        linked = linkify_servicenow_refs(redacted_text, sn_url)
                        await client.chat_update(
                            channel=channel_id,
                            ts=placeholder_ts,
                            text=linked,
                            blocks=blocks,
                        )
                    except Exception:
                        logger.debug(
                            "Failed to update placeholder for timed-out review %d",
                            review_id, exc_info=True,
                        )

                # Create pending recommendation approval
                rec_ids = [r["db_id"] for r in untrusted if r.get("db_id")]
                try:
                    approval_id = await db.create_pending_rec_approval(
                        channel_id=channel_id,
                        message_ts=placeholder_ts,
                        original_text=original,
                        redacted_text=redacted_text,
                        recommendation_ids=rec_ids,
                        thread_ts=thread_ts,
                        interaction_id=review.get("interaction_id"),
                    )

                    # Post standard approval request to #it-helpdesk
                    if settings.it_helpdesk_channel_id:
                        approval_blocks = format_recommendation_approval_blocks(
                            approval_id, untrusted, original,
                            review.get("user_name", "a user"),
                        )
                        resp = await client.chat_postMessage(
                            channel=settings.it_helpdesk_channel_id,
                            text=f"Recommendation approval needed (timed out collaborative review)",
                            blocks=approval_blocks,
                        )
                        await db.update_pending_rec_approval(
                            approval_id,
                            approval_message_ts=resp["ts"],
                            approval_channel=settings.it_helpdesk_channel_id,
                        )
                except Exception:
                    logger.warning(
                        "Failed to create fallback approval for timed-out review %d",
                        review_id, exc_info=True,
                    )
            else:
                # All trusted now — post full response
                if placeholder_ts:
                    try:
                        sn_url = settings.sn_instance_url
                        blocks = format_response_blocks(original, sn_url)
                        linked = linkify_servicenow_refs(original, sn_url)
                        await client.chat_update(
                            channel=channel_id, ts=placeholder_ts,
                            text=linked, blocks=blocks,
                        )
                    except Exception:
                        logger.debug("Failed to update placeholder for review %d", review_id, exc_info=True)
        else:
            # No recommendations found on re-extraction — post original
            if placeholder_ts:
                try:
                    sn_url = settings.sn_instance_url
                    blocks = format_response_blocks(original, sn_url)
                    linked = linkify_servicenow_refs(original, sn_url)
                    await client.chat_update(
                        channel=channel_id, ts=placeholder_ts,
                        text=linked, blocks=blocks,
                    )
                except Exception:
                    logger.debug("Failed to update placeholder for review %d", review_id, exc_info=True)

        # Update the #it-helpdesk message with summary + link
        helpdesk_ts = review.get("helpdesk_message_ts")
        if helpdesk_ts and settings.it_helpdesk_channel_id:
            try:
                summary_line = await _build_collab_decision_line(
                    ":hourglass:", f"timed out after {timeout}min — fell back to standard approval",
                    "system", review, settings,
                )
                await client.chat_update(
                    channel=settings.it_helpdesk_channel_id,
                    ts=helpdesk_ts, text=summary_line,
                    blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": summary_line}}],
                )
            except Exception:
                logger.debug(
                    "Failed to update helpdesk message for timed-out review %d",
                    review_id, exc_info=True,
                )

        await emit("collab_review_timed_out", {"review_id": review_id})


# ---------------------------------------------------------------------------
# KB suggestion for collaborative reviews
# ---------------------------------------------------------------------------

async def _suggest_kb_article_for_collab(
    ticket_id: str, ticket_data: dict, settings: Settings,
) -> None:
    """Check if a resolved ticket had a collaborative review and suggest a KB article."""
    try:
        review = await db.get_collaborative_review_by_ticket(ticket_id)
        if not review:
            return
        if review["status"] == "pending":
            return

        # Collect conversation history for analysis
        history = _collect_ticket_history(ticket_id)
        if not history:
            return

        suggestion = await _generate_kb_suggestion(history, ticket_data, settings)
        if not suggestion or not suggestion.get("worth_creating"):
            return

        # Post KB suggestion in the collaborative review's #it-helpdesk thread
        helpdesk_ts = review.get("helpdesk_message_ts")
        if not helpdesk_ts or not settings.it_helpdesk_channel_id:
            return

        kb_blocks = format_kb_suggestion_blocks(
            review_id=review["id"],
            suggested_title=suggestion.get("title", "Untitled"),
            key_points=suggestion.get("key_points", []),
        )
        client = AsyncWebClient(token=settings.slack_bot_token)
        await client.chat_postMessage(
            channel=settings.it_helpdesk_channel_id,
            thread_ts=helpdesk_ts,
            text=":books: KB Article Suggestion",
            blocks=kb_blocks,
        )
    except Exception:
        logger.debug("Failed to suggest KB article for ticket %s", ticket_id, exc_info=True)


async def _generate_kb_suggestion(
    history: list[dict], ticket_data: dict, settings: Settings,
) -> dict | None:
    """Use Gemini to analyze a resolved conversation and suggest a KB article.

    Returns ``{title, key_points, worth_creating}`` or ``None`` on failure.
    """
    try:
        lines = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "IT Agent"
            content = msg["content"]
            if content.startswith("[System"):
                content = content.split("]\n\n", 1)[-1] if "]\n\n" in content else content
            lines.append(f"{role}: {content}")
        conversation_text = "\n\n".join(lines)

        close_notes = ticket_data.get("close_notes", "")

        client = genai.Client(api_key=settings.gemini_api_key)
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=(
                "Analyze this resolved IT support conversation and determine if it would make "
                "a good knowledge base article. Consider:\n"
                "- Is this a common issue others might face?\n"
                "- Was the resolution clearly identified?\n"
                "- Would documenting the solution save time for future incidents?\n\n"
                f"Resolution notes: {close_notes}\n\n"
                f"Conversation:\n{conversation_text[:3000]}\n\n"
                "Respond with ONLY a JSON object:\n"
                '{"worth_creating": true/false, "title": "suggested article title", '
                '"key_points": ["point1", "point2", "point3"]}\n\n'
                "If not worth creating, set worth_creating to false and leave title/key_points empty."
            ),
        )
        raw = (response.text or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
        return None
    except Exception:
        logger.debug("Failed to generate KB suggestion", exc_info=True)
        return None
