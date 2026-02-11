"""Gemini agent core — tool loop that drives the AI."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from it_agent.agent.executor import execute_tool
from it_agent.agent.tools import TOOLS
from it_agent.bot.events import emit
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
- **KB result quality**: Critically evaluate knowledge base results before sharing them. Only reference \
articles that are genuinely relevant to the user's question. If the results are about a different \
topic (e.g., user asks about monitors but results are about VMs), discard them and either ask \
clarifying questions or provide general guidance from your own knowledge. A low-relevance result \
is worse than no result.
- When creating tickets, extract a clear title and description from the conversation.
- Always confirm actions with the user (e.g., "I've created ticket #5 for your issue").
- If you can't resolve an issue, create a ticket and let the user know.
- Format responses for Slack using markdown (*bold*, `code`, bullet points).
- **INTERNAL articles**: KB articles whose content starts with "INTERNAL" are for internal IT staff \
only. You may use them to inform your troubleshooting, but NEVER share their content, text, or \
procedures directly with the user. Do not quote, paraphrase, or reference the internal content. \
Only share information from non-internal articles.

When responding in the #help-it channel:
- Be proactive: acknowledge the issue immediately and begin troubleshooting.
- Run relevant diagnostics and search the knowledge base without waiting for the user to ask.
- **ALWAYS create a ticket for every new request** — even if you can resolve it immediately via \
the knowledge base. Every interaction must be tracked. Create the ticket as your first action, \
then continue troubleshooting.
- Always include the ticket number and private channel name in your response.
- When the user confirms the issue is resolved (e.g., "that worked", "all good", "this helped", \
"that would be all"), **do NOT immediately resolve the ticket**. First ask what fixed it \
(e.g. "Could you briefly share what resolved the issue so I can add it to the ticket?"). \
Once the user provides details, resolve with those as close_notes. If the user declines or \
is unsure, resolve with a brief summary from conversation context.
"""

# Build Gemini function declarations from our TOOLS list at module level.
GEMINI_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=t["parameters"],
            )
            for t in TOOLS
        ]
    )
]


@dataclass
class AgentResult:
    """Structured result from an agent run."""

    text: str
    tool_calls: list[dict] = field(default_factory=list)
    # Each entry: {"name": "create_ticket", "args": {...}, "result": {...}}


class Agent:
    """Gemini-powered IT support agent with tool use."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model
        self.max_loops = settings.max_tool_loops

    async def run(self, messages: list[dict], user_id: str = "unknown") -> AgentResult:
        """Run the agent tool loop and return a structured AgentResult."""
        await emit("agent_start", {"user_id": user_id, "message_count": len(messages)})

        # Convert incoming messages to Gemini Content objects.
        gemini_contents: list[types.Content] = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            gemini_contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=m["content"])])
            )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=GEMINI_TOOLS,
        )

        all_tool_calls: list[dict] = []

        for loop_idx in range(self.max_loops):
            logger.debug("Agent loop %d, sending %d messages", loop_idx, len(gemini_contents))

            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=gemini_contents,
                config=config,
            )

            # If no function calls, the model is done — return the text.
            if not response.function_calls:
                text = _extract_text(response)
                await emit("agent_response", {
                    "text_preview": text[:200],
                    "tool_call_count": len(all_tool_calls),
                })
                return AgentResult(text=text, tool_calls=all_tool_calls)

            # The model wants to call tools — add its response as a model turn.
            gemini_contents.append(response.candidates[0].content)

            # Execute each function call and collect results.
            function_responses: list[types.Part] = []
            for fc in response.function_calls:
                logger.info("Executing tool: %s(%s)", fc.name, json.dumps(fc.args))
                await emit("tool_call", {"tool_name": fc.name, "args": dict(fc.args) if fc.args else {}})
                result = await execute_tool(fc.name, fc.args, self.settings, user_id)

                result_parsed = json.loads(result)
                all_tool_calls.append({
                    "name": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                    "result": result_parsed,
                })
                result_summary = str(result_parsed)[:200]
                await emit("tool_result", {"tool_name": fc.name, "result_summary": result_summary})

                function_responses.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result},
                    )
                )

            # Add tool results as a user turn.
            gemini_contents.append(types.Content(role="user", parts=function_responses))

        # Safety: max loops reached
        text = (
            "I've reached my processing limit for this request. "
            "Please try breaking your question into smaller parts."
        )
        await emit("agent_response", {
            "text_preview": text[:200],
            "tool_call_count": len(all_tool_calls),
        })
        return AgentResult(text=text, tool_calls=all_tool_calls)


def _extract_text(response) -> str:
    """Extract text content from a Gemini response."""
    if response.text:
        return response.text
    return "I processed your request but have no text to display."
