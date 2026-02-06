from __future__ import annotations

import logging
import re

from slack_bolt.async_app import AsyncApp

from it_agent.agent.core import Agent
from it_agent.bot.formatters import format_error_blocks, format_response_blocks
from it_agent.config import Settings

logger = logging.getLogger(__name__)

# Per-thread conversation history: {(channel, thread_ts): [messages]}
_conversations: dict[tuple[str, str], list[dict]] = {}
MAX_HISTORY = 20

# Shared agent instance
_agent: Agent | None = None


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
        # Strip the bot mention from the text
        text = re.sub(r"<@[A-Z0-9]+>\s*", "", event.get("text", "")).strip()
        if not text:
            await say("Hi! I'm the IT Support Agent. How can I help you?")
            return
        await _handle_message(event, text, say, settings)

    @app.event("message")
    async def handle_dm(event: dict, say) -> None:
        """Handle direct messages."""
        # Ignore bot messages, edits, etc.
        if event.get("subtype"):
            return
        text = event.get("text", "").strip()
        if not text:
            return
        await _handle_message(event, text, say, settings)


async def _handle_message(event: dict, text: str, say, settings: Settings) -> None:
    """Process a user message through the agent."""
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
        response = await agent.run(history, user_id=user_id)

        # Append assistant response to history
        history.append({"role": "assistant", "content": response})

        blocks = format_response_blocks(response)
        await say(text=response, blocks=blocks, thread_ts=thread_ts)

    except Exception:
        logger.exception("Error processing message")
        blocks = format_error_blocks(
            "Something went wrong processing your request. Please try again."
        )
        await say(text="Error processing request", blocks=blocks, thread_ts=thread_ts)
