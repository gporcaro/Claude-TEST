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

    async def _assignment_callback(
        ticket_id: str, assignee_id: str, settings: Settings
    ) -> None:
        await _handle_ticket_assigned(ticket_id, assignee_id, settings)

    executor._on_ticket_assigned = _assignment_callback


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

    # Recover history from Slack if we have none (e.g. after bot restart)
    if conv_key not in _conversations:
        recovered = await _recover_thread_history(channel, thread_ts, settings)
        if recovered:
            # The last message in recovered history is the current message,
            # so we use the recovered history directly
            _conversations[conv_key] = recovered
            history = _conversations[conv_key]
        else:
            _conversations[conv_key] = [{"role": "user", "content": text}]
            history = _conversations[conv_key]
    else:
        history = _conversations[conv_key]
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

        # Forward thread replies to the incident channel
        if event.get("thread_ts"):
            try:
                inc_channel = _find_incident_channel_for_thread(channel, thread_ts)
                if inc_channel:
                    slack = AsyncWebClient(token=settings.slack_bot_token)
                    # Resolve display name for attribution
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
            except Exception:
                logger.debug("Failed to forward thread reply to incident channel", exc_info=True)

        # Collect KB results from this run
        kb_results: list[dict] = []
        for tc in result.tool_calls:
            if tc["name"] == "search_knowledge_base":
                kb_results.extend(tc.get("result", {}).get("results", []))

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
            channel_id = r.get("channel_id", "")

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

            # Post a permalink to the original #help-it thread in the incident channel
            if channel_id:
                try:
                    slack = AsyncWebClient(token=settings.slack_bot_token)
                    plink = await slack.chat_getPermalink(
                        channel=channel, message_ts=thread_ts,
                    )
                    permalink = plink.get("permalink", "")
                    if permalink:
                        await slack.chat_postMessage(
                            channel=channel_id,
                            text=f":link: <{permalink}|Original #help-it thread>",
                        )
                except Exception:
                    logger.debug(
                        "Failed to post thread permalink in incident channel %s",
                        channel_id, exc_info=True,
                    )

            # Post KB article content in the incident channel
            if kb_results and channel_id:
                await _post_kb_results_to_channel(
                    channel_id, kb_results, settings,
                )

    except Exception:
        logger.exception("Error processing #help-it message")
        blocks = format_error_blocks(
            "Something went wrong processing your request. Please try again."
        )
        await say(text="Error processing request", blocks=blocks, thread_ts=thread_ts)


async def _post_kb_results_to_channel(
    channel_id: str, kb_results: list[dict], settings: Settings,
) -> None:
    """Post KB article references and content in the incident channel."""
    sn_url = settings.sn_instance_url
    client = AsyncWebClient(token=settings.slack_bot_token)

    for article in kb_results:
        article_id = article.get("id", "")
        title = article.get("title", "Untitled")
        content = article.get("content", "")
        source = article.get("source", "")

        # Skip internal articles — they can inform the agent but shouldn't
        # be shared directly with users.
        if content.strip().upper().startswith("INTERNAL"):
            continue

        # Build the message
        if article_id.startswith("KB"):
            header = f":book: *{article_id} — {title}*"
        elif source == "local":
            header = f":book: *{title}*"
        else:
            header = f":book: *{article_id} — {title}*"

        message = f"{header}\n\n{content}"
        message = linkify_servicenow_refs(message, sn_url)

        try:
            await client.chat_postMessage(channel=channel_id, text=message)
        except Exception:
            logger.warning(
                "Failed to post KB article %s to channel %s", article_id, channel_id,
                exc_info=True,
            )


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


async def _handle_ticket_assigned(
    ticket_id: str, assignee_sn_sys_id: str, settings: Settings
) -> None:
    """Invite the assigned user to the incident Slack channel and post a notification."""
    # Reverse-lookup: find channel_id whose context has this ticket_id
    channel_id: str | None = None
    for ch_id, ctx in _incident_context.items():
        if ctx.get("ticket_id") == ticket_id:
            channel_id = ch_id
            break

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
