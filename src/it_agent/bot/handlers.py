from __future__ import annotations

import logging
import re

from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from it_agent.agent import executor
from it_agent.agent.core import Agent, AgentResult
from it_agent.bot.formatters import format_error_blocks, format_response_blocks
from it_agent.config import Settings

logger = logging.getLogger(__name__)

# Per-thread conversation history: {(channel, thread_ts): [messages]}
_conversations: dict[tuple[str, str], list[dict]] = {}
MAX_HISTORY = 20

# Shared agent instance
_agent: Agent | None = None

# Tracks ticket → original #help-it thread so we can post resolution updates.
# ticket_id → (channel, thread_ts)
_ticket_threads: dict[str, tuple[str, str]] = {}


def _get_agent(settings: Settings) -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent(settings)
    return _agent


def register_handlers(app: AsyncApp, settings: Settings) -> None:
    """Register Slack event handlers."""

    @app.event("app_mention")
    async def handle_mention(event: dict, say) -> None:
        """Handle @bot mentions in channels."""
        # If the mention is in #help-it, skip — handle_message covers it.
        if settings.help_channel_id and event.get("channel") == settings.help_channel_id:
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

        # Other channels → ignore (handled by app_mention only)

    # Wire the resolution callback
    async def _resolution_callback(
        ticket_id: str, ticket_data: dict, settings: Settings
    ) -> None:
        await post_resolution_update(ticket_id, ticket_data, settings)

    executor._on_ticket_resolved = _resolution_callback


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

        blocks = format_response_blocks(result.text)
        await say(text=result.text, blocks=blocks, thread_ts=thread_ts)

    except Exception:
        logger.exception("Error processing message")
        blocks = format_error_blocks(
            "Something went wrong processing your request. Please try again."
        )
        await say(text="Error processing request", blocks=blocks, thread_ts=thread_ts)


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

        blocks = format_response_blocks(result.text)
        await say(text=result.text, blocks=blocks, thread_ts=thread_ts)

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
    """Post a resolution message to the original #help-it thread."""
    thread_info = _ticket_threads.pop(ticket_id, None)
    if thread_info is None:
        return

    channel, thread_ts = thread_info
    status = ticket_data.get("status", "resolved")
    title = ticket_data.get("title", ticket_id)
    close_notes = ticket_data.get("close_notes", "")

    message = f":white_check_mark: *Ticket {ticket_id} — {status}.* {title}"
    if close_notes:
        message += f"\n>_{close_notes}_"

    client = AsyncWebClient(token=settings.slack_bot_token)
    await client.chat_postMessage(channel=channel, text=message, thread_ts=thread_ts)
