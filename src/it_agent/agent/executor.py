"""Tool name → function dispatch."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

from it_agent.config import Settings
from it_agent.tools.diagnostics import check_disk_usage, check_service_status, dns_lookup, ping_host
from it_agent.tools.knowledge import search_knowledge_base
from it_agent.tools.tickets import create_ticket, get_ticket, list_tickets, update_ticket

logger = logging.getLogger(__name__)

# Registry mapping tool names to async handler functions
_TOOL_HANDLERS = {
    "ping_host": ping_host,
    "dns_lookup": dns_lookup,
    "check_disk_usage": check_disk_usage,
    "check_service_status": check_service_status,
    "create_ticket": create_ticket,
    "get_ticket": get_ticket,
    "update_ticket": update_ticket,
    "list_tickets": list_tickets,
    "search_knowledge_base": search_knowledge_base,
}

# Resolution callback — set by handlers.py to post updates back to #help-it.
# Signature: async (ticket_id: str, ticket_data: dict, settings: Settings) -> None
_on_ticket_resolved: Callable | None = None


async def execute_tool(
    tool_name: str, tool_input: dict, settings: Settings, user_id: str = "unknown"
) -> str:
    """Execute a tool by name and return the JSON result string."""
    handler = _TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        # Work on a mutable copy — fc.args from GenAI SDK may be immutable
        args = dict(tool_input) if tool_input else {}

        # Inject settings and user_id for tools that need them
        if tool_name in ("create_ticket", "update_ticket"):
            args["_settings"] = settings
            args["_user_id"] = user_id
        elif tool_name in ("get_ticket", "list_tickets"):
            args["_settings"] = settings
        elif tool_name == "search_knowledge_base":
            args["_settings"] = settings

        result = await handler(**args)
        result_str = json.dumps(result, default=str)

        # Fire resolution callback when a ticket is resolved or closed
        if (
            tool_name == "update_ticket"
            and result.get("success")
            and _on_ticket_resolved is not None
        ):
            status = args.get("status", "")
            if status in ("resolved", "closed"):
                try:
                    ticket_id = args.get("ticket_id", "")
                    ticket_data = result.get("ticket", {})
                    await _on_ticket_resolved(ticket_id, ticket_data, settings)
                except Exception:
                    logger.exception(
                        "Resolution callback failed for %s", tool_input.get("ticket_id")
                    )

        return result_str
    except Exception as e:
        logger.exception("Tool execution error: %s", tool_name)
        return json.dumps({"error": str(e)})
