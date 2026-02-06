"""Tool name → function dispatch."""

from __future__ import annotations

import json
import logging

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


async def execute_tool(
    tool_name: str, tool_input: dict, settings: Settings, user_id: str = "unknown"
) -> str:
    """Execute a tool by name and return the JSON result string."""
    handler = _TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        # Inject settings and user_id for tools that need them
        if tool_name in ("create_ticket", "update_ticket"):
            tool_input["_settings"] = settings
            tool_input["_user_id"] = user_id
        elif tool_name in ("get_ticket", "list_tickets"):
            tool_input["_settings"] = settings
        elif tool_name == "search_knowledge_base":
            tool_input["_settings"] = settings

        result = await handler(**tool_input)
        return json.dumps(result, default=str)
    except Exception as e:
        logger.exception("Tool execution error: %s", tool_name)
        return json.dumps({"error": str(e)})
