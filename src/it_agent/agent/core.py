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

Company environment:
- ~80% of employees use MacBooks (various models), ~20% use Dell Windows laptops.
- When a user asks about hardware (monitors, docking stations, peripherals, etc.) and doesn't \
specify their device, assume it's likely a MacBook but ask to confirm. Tailor your guidance \
to MacBook (USB-C/Thunderbolt) or Dell (USB-C/HDMI/DisplayPort) accordingly.

Your capabilities:
- **Diagnostics**: Ping hosts, DNS lookups, check disk usage, check service status. \
IMPORTANT: These diagnostic tools run on the IT infrastructure server, NOT on the user's \
device. `check_disk_usage` reports the server's disk, not the user's laptop. `ping_host` \
and `dns_lookup` test connectivity from the server. Never present server-side diagnostic \
results as if they came from the user's machine. For user device issues (slow laptop, disk \
full, etc.), guide the user through checking their own system (e.g., Task Manager, Activity \
Monitor, System Settings) rather than running server-side diagnostics.
- **Ticket Management**: Create, view, update, and list IT support tickets
- **Knowledge Base**: Search internal IT documentation for solutions and procedures
- **Public Articles**: Search official vendor documentation (Apple, Microsoft, Dell) for \
device-specific and OS-related issues. Use `search_public_articles` when the internal KB \
doesn't have relevant results. Articles have confidence scores from user feedback — prefer \
higher-confidence articles. Always include the source URL when sharing public articles.

Guidelines:
- Be helpful, concise, and professional. Address the user by their first name when possible.
- When a user reports an issue, try to diagnose it first using diagnostic tools before escalating.
- Search the knowledge base for common issues before creating tickets.
- **KB result quality**: Critically evaluate knowledge base results before sharing them. Only reference \
articles that are genuinely relevant to the user's question. If the results are about a different \
topic (e.g., user asks about monitors but results are about VMs), discard them and either ask \
clarifying questions or provide general guidance from your own knowledge. A low-relevance result \
is worse than no result.
- **Never tell the user** that you searched the knowledge base and found nothing. If no relevant \
results are found, silently move on — provide troubleshooting guidance from your own knowledge \
or ask clarifying questions. Saying "I couldn't find anything in our KB" makes the company look \
under-resourced.
- **Laptop performance / slowness**: Before escalating any laptop performance issue, always ask \
the user when they last rebooted. If the last reboot was more than a week ago, ask them to \
restart the laptop and check if the issue persists before taking further action. Many performance \
issues are resolved by a simple reboot.
- When creating tickets, extract a clear title and description from the conversation.
- Always confirm actions with the user (e.g., "I've created ticket #5 for your issue").
- If you can't resolve an issue, create a ticket and let the user know.
- When the internal knowledge base has no relevant results for hardware or OS questions, \
search public articles from official vendor sources before giving general advice.
- For public articles: share the title, a brief summary, and the URL. Do not copy-paste \
the full article content — just guide the user to the relevant resource.
- **Ticket accuracy**: When you call `update_ticket` to change priority, status, or other fields, \
always check the tool result to confirm the update succeeded. Only state the new value if \
the result confirms it. Never claim a ticket is "high priority" if the update failed or \
returned a different value.
- **Escalation language**: Never use the word "human" when referring to escalation or transfer. \
Instead say "Support Agent", "Support Representative", or "next level of support".
- Format responses for Slack using markdown (*bold*, `code`, bullet points).
- **INTERNAL articles**: KB articles whose content starts with "INTERNAL" are for internal IT staff \
only. You may use them to inform your troubleshooting, but NEVER share their content, text, or \
procedures directly with the user. Do not quote, paraphrase, or reference the internal content. \
Only share information from non-internal articles.
- **[AI Context] articles**: You may be provided with authoritative context articles below. These \
are curated by IT staff and contain verified information about applications, access procedures, \
troubleshooting flows, and escalation paths. ALWAYS prioritize information from these articles \
over general KB search results or your own training knowledge. Use them proactively when the \
user's question relates to a covered topic — do not search the KB for topics already covered \
by an [AI Context] article.
- **Admin-side checks**: Never ask the user to verify things they cannot check or do not have \
access to. Okta group memberships, Active Directory groups, license assignments, SSO \
configurations, firewall rules, and application provisioning are all IT-managed. If access \
to an app depends on group membership or license assignment, either check it yourself using \
available tools or note it as something IT will verify — do not tell the user to "check if \
you are a member of" an Okta/AD group or to "ensure your license is assigned".

When responding in the #help-it channel:
- Be proactive: acknowledge the issue immediately and begin troubleshooting.
- Run relevant diagnostics and search the knowledge base without waiting for the user to ask.
- **Tickets are created automatically** — do NOT call `create_ticket` yourself in #help-it. \
The ticket ID is provided in conversation context. Reference it in your response.
- No private channel is created initially — the conversation continues in the #help-it thread. \
A channel will be created only if the user requests it or the ticket is escalated. \
Do NOT mention a private channel unless one has been created.
- **Escalation**: When the user asks for a real person, to escalate, or to be transferred, \
you MUST call `update_ticket` with `priority="high"` and a `comment` noting the escalation \
request — even if you already raised the priority before. This triggers the system to create \
a private channel and queue the ticket for assignment. Do NOT say a technician "has been \
notified" or "will contact" the user unless the ticket has actually been assigned to a \
specific person via `assignee_id`. Be honest: say "I've escalated the ticket — it's now in \
the queue for a Support Agent."
When responding in an incident channel (private channels named like inc0129540-username):
- You are ALREADY in a private troubleshooting channel. Do NOT mention creating a private channel \
or say a Support Agent will "reach out in a private channel" — the user is already here.
- **Escalation**: When the user asks for a real person or to escalate, call `update_ticket` with \
`priority="high"` and a `comment` noting the escalation request. Then tell the user: "I've \
escalated the ticket to high priority. IT staff have been notified in #it-helpdesk and will \
join this channel to assist you." Do NOT promise someone will "reach out" or "contact you in \
a private channel" — they are already in the channel.

- When the user confirms the issue is resolved (e.g., "that worked", "all good", "this helped", \
"that would be all"), **do NOT immediately resolve the ticket**. First, carefully analyze the \
user's message and the conversation history to determine if the resolution reason has already \
been provided (e.g., "I resolved this by rebooting", "a restart fixed it", "I replaced the \
cable and it works now"). If the user already explained what fixed it — either in the current \
message or earlier in the conversation — use that as the close_notes and resolve the ticket \
immediately without asking again. Only ask "Could you briefly share what resolved the issue?" \
if the resolution cause is genuinely unclear (e.g., the user just says "it's working now" \
with no explanation). If the user declines or is unsure, resolve with a brief summary from \
conversation context.
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

    async def run(
        self,
        messages: list[dict],
        user_id: str = "unknown",
        context_articles: list[dict] | None = None,
    ) -> AgentResult:
        """Run the agent tool loop and return a structured AgentResult."""
        await emit("agent_start", {"user_id": user_id, "message_count": len(messages)})

        # Convert incoming messages to Gemini Content objects.
        gemini_contents: list[types.Content] = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            gemini_contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=m["content"])])
            )

        # Build system prompt, injecting [AI Context] articles when available.
        system_prompt = SYSTEM_PROMPT
        if context_articles:
            sections = []
            for art in context_articles:
                sections.append(
                    f"### {art['title']} ({art['id']})\n{art['content']}"
                )
            system_prompt += (
                "\n\n---\n## Authoritative AI Context Articles\n"
                "The following articles are curated by IT staff. "
                "Treat them as your primary source of truth.\n\n"
                + "\n\n".join(sections)
            )

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
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
