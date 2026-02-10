"""Tests for tool schema definitions."""

from __future__ import annotations

from it_agent.agent.tools import TOOLS


def test_all_tools_present():
    names = {t["name"] for t in TOOLS}
    expected = {
        "ping_host",
        "dns_lookup",
        "check_disk_usage",
        "check_service_status",
        "create_ticket",
        "get_ticket",
        "update_ticket",
        "list_tickets",
        "search_knowledge_base",
    }
    assert names == expected


def test_all_tools_have_required_fields():
    for tool in TOOLS:
        assert "name" in tool
        assert "description" in tool
        assert "parameters" in tool
        assert tool["parameters"]["type"] == "object"
        assert "properties" in tool["parameters"]


def test_ticket_id_is_string():
    """ticket_id should be a string for ServiceNow incident numbers."""
    tools_by_name = {t["name"]: t for t in TOOLS}

    get_tool = tools_by_name["get_ticket"]
    assert get_tool["parameters"]["properties"]["ticket_id"]["type"] == "string"

    update_tool = tools_by_name["update_ticket"]
    assert update_tool["parameters"]["properties"]["ticket_id"]["type"] == "string"
