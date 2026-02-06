"""Claude agent core — tool loop that drives the AI."""

from __future__ import annotations

import json
import logging

import anthropic

from it_agent.agent.executor import execute_tool
from it_agent.agent.tools import TOOLS
from it_agent.config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an IT Support Agent. You help employees with technical issues, manage support tickets, \
and search the company knowledge base for solutions.

Your capabilities:
- **Diagnostics**: Ping hosts, DNS lookups, check disk usage, check service status
- **Ticket Management**: Create, view, update, and list IT support tickets
- **Knowledge Base**: Search internal IT documentation for solutions and procedures

Guidelines:
- Be helpful, concise, and professional.
- When a user reports an issue, try to diagnose it first using diagnostic tools before escalating.
- Search the knowledge base for common issues before creating tickets.
- When creating tickets, extract a clear title and description from the conversation.
- Always confirm actions with the user (e.g., "I've created ticket #5 for your issue").
- If you can't resolve an issue, create a ticket and let the user know.
- Format responses for Slack using markdown (*bold*, `code`, bullet points).
"""


class Agent:
    """Claude-powered IT support agent with tool use."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_model
        self.max_loops = settings.max_tool_loops

    async def run(self, messages: list[dict], user_id: str = "unknown") -> str:
        """Run the agent tool loop and return the final text response."""
        # Build messages for Claude (only role + content)
        claude_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

        for loop_idx in range(self.max_loops):
            logger.debug("Agent loop %d, sending %d messages", loop_idx, len(claude_messages))

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=claude_messages,
            )

            # If the model wants to stop, extract text and return
            if response.stop_reason == "end_turn":
                return _extract_text(response)

            # If the model wants to use tools, execute them
            if response.stop_reason == "tool_use":
                # Add the assistant message with all content blocks
                claude_messages.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                    }
                )

                # Execute each tool call and build tool results
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        logger.info("Executing tool: %s(%s)", block.name, json.dumps(block.input))
                        result = await execute_tool(block.name, block.input, self.settings, user_id)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )

                claude_messages.append({"role": "user", "content": tool_results})
            else:
                # Unexpected stop reason, return whatever text we have
                return _extract_text(response)

        # Safety: max loops reached
        return (
            "I've reached my processing limit for this request. "
            "Please try breaking your question into smaller parts."
        )


def _extract_text(response) -> str:
    """Extract text content from a Claude response."""
    parts = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts) if parts else "I processed your request but have no text to display."
