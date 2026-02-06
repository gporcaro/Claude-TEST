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
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"
        assert "properties" in tool["input_schema"]
