from __future__ import annotations

import asyncio
import logging
import re
import time

from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from it_agent.agent import executor
from it_agent.agent.core import Agent, AgentResult
from it_agent.bot.formatters import (
    format_error_blocks,
    format_response_blocks,
    linkify_servicenow_refs,
)
from it_agent.config import Settings
from it_agent.servicenow.client import ServiceNowClient

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

# Resolved tickets pending auto-close: ticket_id → resolved_epoch
_resolved_pending_close: dict[str, float] = {}

# 48 hours in seconds
_AUTO_CLOSE_DELAY = 48 * 60 * 60

# How often to check (1 hour)
_AUTO_CLOSE_CHECK_INTERVAL = 60 * 60


def _get_agent(settings: Settings) -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent(settings)
    return _agent


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

        # Fetch context from the first bot message in each channel
        for ch_id in channel_ids:
            try:
                hist = await client.conversations_history(channel=ch_id, limit=50)
                # Scan all messages for the incident summary (posted by the bot)
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
                        break
            except Exception:
                logger.debug("Could not fetch history for channel %s", ch_id)

        if _incident_channels:
            logger.info(
                "Discovered %d incident channel(s) on startup (%d with context)",
                len(_incident_channels),
                len(_incident_context),
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


def _register_incident_channels(result: AgentResult) -> None:
    """Track any incident channels created during this agent run."""
    for tc in result.tool_calls:
        if tc["name"] != "create_ticket":
            continue
        r = tc.get("result", {})
        channel_id = r.get("channel_id")
        if channel_id:
            _incident_channels.add(channel_id)
            ticket = r.get("ticket", {})
            _incident_context[channel_id] = {
                "ticket_id": ticket.get("ticket_id", ""),
                "title": ticket.get("title", ""),
                "description": ticket.get("description", ""),
                "priority": ticket.get("priority", ""),
                "summary_ts": r.get("summary_ts"),
                "original_text": None,  # will be built on first summary update
                "summary_lines": [],
            }
            logger.info("Registered incident channel %s", channel_id)


async def _handle_message(event: dict, text: str, say, settings: Settings) -> None:
    """Process a user message through the agent (DM / mention flow)."""
    channel = event["channel"]
    thread_ts = event.get("thread_ts") or event["ts"]
    user_id = event.get("user", "unknown")

    # Build conversation key
    conv_key = (channel, thread_ts)

    # Get or init history
    history = _conversations.setdefault(conv_key, [])
    history.append({"role": "user", "content": text})

    # Trim old history
    if len(history) > MAX_HISTORY:
        _conversations[conv_key] = history[-MAX_HISTORY:]
        history = _conversations[conv_key]

    try:
        agent = _get_agent(settings)
        result: AgentResult = await agent.run(history, user_id=user_id)

        # Append assistant response to history
        history.append({"role": "assistant", "content": result.text})

        _register_incident_channels(result)

        sn_url = settings.sn_instance_url
        linked_text = linkify_servicenow_refs(result.text, sn_url)
        blocks = format_response_blocks(result.text, sn_url)
        await say(text=linked_text, blocks=blocks, thread_ts=thread_ts)

    except Exception:
        logger.exception("Error processing message")
        blocks = format_error_blocks(
            "Something went wrong processing your request. Please try again."
        )
        await say(text="Error processing request", blocks=blocks, thread_ts=thread_ts)


async def _handle_incident_message(
    event: dict, text: str, say, settings: Settings
) -> None:
    """Process a message in an incident channel — no threading, single conversation.

    Seeds the conversation with incident context on first interaction and
    updates the channel's pinned summary message after each agent turn.
    """
    channel = event["channel"]
    user_id = event.get("user", "unknown")

    # Single conversation per incident channel (no per-thread splitting)
    conv_key = (channel, "incident")
    history = _conversations.setdefault(conv_key, [])

    # Seed context on first interaction so the agent knows what the channel is about
    if not history:
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
            history.append({"role": "user", "content": context_msg})
            history.append({
                "role": "assistant",
                "content": (
                    f"Understood. I'm tracking incident {ctx.get('ticket_id', 'this issue')}: "
                    f"\"{ctx.get('title', '')}\". "
                    f"I'll help troubleshoot. What would you like me to do?"
                ),
            })

    history.append({"role": "user", "content": text})

    if len(history) > MAX_HISTORY:
        _conversations[conv_key] = history[-MAX_HISTORY:]
        history = _conversations[conv_key]

    try:
        agent = _get_agent(settings)
        result: AgentResult = await agent.run(history, user_id=user_id)

        history.append({"role": "assistant", "content": result.text})

        _register_incident_channels(result)

        sn_url = settings.sn_instance_url
        linked_text = linkify_servicenow_refs(result.text, sn_url)
        blocks = format_response_blocks(result.text, sn_url)
        await say(text=linked_text, blocks=blocks)

        # Update the pinned summary message with progress
        await _update_incident_summary(channel, result.text, settings)

    except Exception:
        logger.exception("Error processing incident channel message")
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


async def _handle_help_channel_message(
    event: dict, text: str, say, settings: Settings
) -> None:
    """Handle a message in #help-it — same as _handle_message but with ticket follow-up."""
    channel = event["channel"]
    thread_ts = event.get("thread_ts") or event["ts"]
    user_id = event.get("user", "unknown")

    conv_key = (channel, thread_ts)
    history = _conversations.setdefault(conv_key, [])
    history.append({"role": "user", "content": text})

    if len(history) > MAX_HISTORY:
        _conversations[conv_key] = history[-MAX_HISTORY:]
        history = _conversations[conv_key]

    try:
        agent = _get_agent(settings)
        result: AgentResult = await agent.run(history, user_id=user_id)

        history.append({"role": "assistant", "content": result.text})

        _register_incident_channels(result)

        sn_url = settings.sn_instance_url
        linked_text = linkify_servicenow_refs(result.text, sn_url)
        blocks = format_response_blocks(result.text, sn_url)
        await say(text=linked_text, blocks=blocks, thread_ts=thread_ts)

        # Post threaded follow-ups for any tickets created during this run
        for tc in result.tool_calls:
            if tc["name"] != "create_ticket":
                continue
            r = tc.get("result", {})
            if not r.get("success"):
                continue

            ticket = r.get("ticket", {})
            ticket_id = ticket.get("ticket_id", "")
            channel_name = r.get("channel_name", "")

            if not ticket_id:
                continue

            # Avoid duplicate follow-ups for the same ticket
            if ticket_id in _ticket_threads:
                continue

            _ticket_threads[ticket_id] = (channel, thread_ts)

            followup = (
                f":ticket: *Ticket {ticket_id} created.* "
                f"A private channel *#{channel_name}* has been created "
                f"to troubleshoot this issue. Updates will be posted here "
                f"after resolution."
            )
            followup = linkify_servicenow_refs(followup, sn_url)
            await say(text=followup, thread_ts=thread_ts)

    except Exception:
        logger.exception("Error processing #help-it message")
        blocks = format_error_blocks(
            "Something went wrong processing your request. Please try again."
        )
        await say(text="Error processing request", blocks=blocks, thread_ts=thread_ts)


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

    # Rename the incident channel to append "-resolved"
    await _rename_incident_channel_resolved(ticket_id, client)

    # Schedule auto-close after 48 hours
    _resolved_pending_close[ticket_id] = time.time()
    logger.info("Ticket %s queued for auto-close in 48h", ticket_id)


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
        await asyncio.sleep(_AUTO_CLOSE_CHECK_INTERVAL)
        try:
            await _process_pending_auto_closes(settings)
        except Exception:
            logger.warning("Auto-close loop iteration failed", exc_info=True)


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

    # Clean up tracking
    _incident_channels.discard(channel_id)
    _incident_context.pop(channel_id, None)
