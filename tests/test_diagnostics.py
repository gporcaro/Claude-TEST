"""Tests for diagnostic tools."""

from __future__ import annotations

import pytest

from it_agent.tools.diagnostics import (
    _validate_host,
    check_disk_usage,
    check_service_status,
    dns_lookup,
    ping_host,
)


class TestValidateHost:
    def test_valid_hostname(self):
        assert _validate_host("google.com") == "google.com"

    def test_valid_ip(self):
        assert _validate_host("192.168.1.1") == "192.168.1.1"

    def test_strips_whitespace(self):
        assert _validate_host("  google.com  ") == "google.com"

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            _validate_host("")

    def test_rejects_shell_injection(self):
        with pytest.raises(ValueError):
            _validate_host("google.com; rm -rf /")

    def test_rejects_backticks(self):
        with pytest.raises(ValueError):
            _validate_host("`whoami`")

    def test_rejects_too_long(self):
        with pytest.raises(ValueError):
            _validate_host("a" * 254)


@pytest.mark.asyncio
async def test_ping_host_localhost():
    result = await ping_host("127.0.0.1", count=1)
    assert result["host"] == "127.0.0.1"
    assert result["status"] in ("reachable", "unreachable", "error")


@pytest.mark.asyncio
async def test_ping_host_invalid():
    result = await ping_host("this.host.definitely.does.not.exist.invalid", count=1)
    assert result["status"] in ("unreachable", "error")


@pytest.mark.asyncio
async def test_dns_lookup():
    result = await dns_lookup("google.com")
    assert result["hostname"] == "google.com"
    assert "addresses" in result or "error" in result


@pytest.mark.asyncio
async def test_check_disk_usage():
    result = await check_disk_usage("/")
    assert result["path"] == "/"
    assert "total_gb" in result
    assert "percent_used" in result
    assert result["status"] in ("ok", "warning", "critical")


@pytest.mark.asyncio
async def test_check_disk_usage_sanitizes_path():
    result = await check_disk_usage("/../../../etc")
    # Should be sanitized to /
    assert result["path"] == "/"


@pytest.mark.asyncio
async def test_check_service_status():
    # Check for a process that's very likely running
    result = await check_service_status("python")
    assert result["service"] == "python"
    assert result["status"] in ("running", "not_running")


@pytest.mark.asyncio
async def test_check_service_rejects_bad_name():
    result = await check_service_status("bad; rm -rf /")
    assert "error" in result
