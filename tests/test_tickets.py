"""Tests for ServiceNow ticket integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from it_agent.config import Settings
from it_agent.servicenow.client import ServiceNowClient, _normalize_incident
from it_agent.tools.tickets import create_ticket

# --- Helpers ---

def _make_sn_incident(**overrides) -> dict:
    """Return a minimal ServiceNow incident record."""
    record = {
        "sys_id": "abc123",
        "number": "INC0010001",
        "short_description": "VPN not working",
        "description": "Cannot connect to VPN",
        "urgency": "2",
        "state": "1",
        "category": "network",
        "caller_id": "U123",
        "assigned_to": "",
        "sys_created_on": "2025-01-01 00:00:00",
        "sys_updated_on": "2025-01-01 00:00:00",
    }
    record.update(overrides)
    return record


def _mock_response(json_body: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.raise_for_status.return_value = None
    return resp


# --- Normalize ---

def test_normalize_incident_maps_fields():
    raw = _make_sn_incident()
    result = _normalize_incident(raw)
    assert result["ticket_id"] == "INC0010001"
    assert result["title"] == "VPN not working"
    assert result["priority"] == "medium"
    assert result["status"] == "open"


def test_normalize_incident_high_urgency():
    raw = _make_sn_incident(urgency="1")
    result = _normalize_incident(raw)
    assert result["priority"] == "high"


def test_normalize_incident_resolved_state():
    raw = _make_sn_incident(state="6")
    result = _normalize_incident(raw)
    assert result["status"] == "resolved"


# --- Client methods ---

@pytest.mark.asyncio
async def test_create_incident():
    client = ServiceNowClient("https://test.service-now.com", "user", "pass")
    mock_resp = _mock_response({"result": _make_sn_incident()})

    mock_post = AsyncMock(return_value=mock_resp)
    with patch.object(client._client, "post", mock_post):
        result = await client.create_incident(
            {"title": "VPN not working", "description": "Cannot connect", "priority": "medium"}
        )
        mock_post.assert_called_once()
        assert result["ticket_id"] == "INC0010001"
        assert result["title"] == "VPN not working"

    await client.close()


@pytest.mark.asyncio
async def test_get_incident():
    client = ServiceNowClient("https://test.service-now.com", "user", "pass")
    mock_resp = _mock_response({"result": [_make_sn_incident()]})

    mock_get = AsyncMock(return_value=mock_resp)
    with patch.object(client._client, "get", mock_get):
        result = await client.get_incident("INC0010001")
        mock_get.assert_called_once()
        assert result is not None
        assert result["ticket_id"] == "INC0010001"

    await client.close()


@pytest.mark.asyncio
async def test_get_incident_not_found():
    client = ServiceNowClient("https://test.service-now.com", "user", "pass")
    mock_resp = _mock_response({"result": []})

    mock_get = AsyncMock(return_value=mock_resp)
    with patch.object(client._client, "get", mock_get):
        result = await client.get_incident("INC9999999")
        assert result is None

    await client.close()


@pytest.mark.asyncio
async def test_update_incident():
    client = ServiceNowClient("https://test.service-now.com", "user", "pass")
    updated_record = _make_sn_incident(state="2")
    mock_resp = _mock_response({"result": updated_record})

    mock_patch = AsyncMock(return_value=mock_resp)
    with patch.object(client._client, "patch", mock_patch):
        result = await client.update_incident("abc123", {"status": "in_progress"})
        mock_patch.assert_called_once()
        assert result["status"] == "in_progress"

    await client.close()


@pytest.mark.asyncio
async def test_list_incidents():
    client = ServiceNowClient("https://test.service-now.com", "user", "pass")
    records = [_make_sn_incident(number=f"INC001000{i}") for i in range(3)]
    mock_resp = _mock_response({"result": records})

    mock_get = AsyncMock(return_value=mock_resp)
    with patch.object(client._client, "get", mock_get):
        results = await client.list_incidents(limit=3)
        mock_get.assert_called_once()
        assert len(results) == 3

    await client.close()


@pytest.mark.asyncio
async def test_create_incident_priority_mapping():
    """Verify that bot priority values map to correct urgency."""
    client = ServiceNowClient("https://test.service-now.com", "user", "pass")
    mock_resp = _mock_response({"result": _make_sn_incident(urgency="1")})

    mock_post = AsyncMock(return_value=mock_resp)
    with patch.object(client._client, "post", mock_post):
        await client.create_incident({"title": "Critical issue", "priority": "critical"})
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["urgency"] == "1"
        assert payload["impact"] == "1"

    await client.close()


# --- create_ticket with Slack channel ---

def _fake_settings() -> Settings:
    return Settings(
        slack_bot_token="xoxb-fake",
        slack_app_token="xapp-fake",
        gemini_api_key="fake-key",
        sn_instance_url="https://test.service-now.com",
        sn_username="user",
        sn_password="pass",
    )


def _normalized_incident(**overrides) -> dict:
    """Return a normalized incident dict as returned by ServiceNowClient."""
    inc = {
        "sys_id": "abc123",
        "ticket_id": "INC0010001",
        "title": "VPN not working",
        "description": "Cannot connect to VPN",
        "priority": "medium",
        "status": "open",
        "category": "network",
        "caller_id": "U123",
        "assigned_to": "",
        "created_at": "2025-01-01 00:00:00",
        "updated_at": "2025-01-01 00:00:00",
    }
    inc.update(overrides)
    return inc


@pytest.mark.asyncio
async def test_create_ticket_creates_slack_channel():
    """Verify that create_ticket creates a private Slack channel and invites the user."""
    settings = _fake_settings()
    incident = _normalized_incident()

    mock_slack = AsyncMock()
    mock_slack.users_info.return_value = {
        "user": {
            "name": "gporcaro",
            "profile": {"email": "gporcaro@test.com", "real_name": "G Porcaro"},
        }
    }
    mock_slack.conversations_create.return_value = {"channel": {"id": "C999"}}
    mock_slack.conversations_invite.return_value = {"ok": True}
    mock_slack.chat_postMessage.return_value = {"ok": True, "ts": "1234567890.123456"}

    # Mock SN user lookup response
    mock_sn_user_resp = MagicMock()
    mock_sn_user_resp.json.return_value = {"result": [{"sys_id": "sn_user_123"}]}
    mock_sn_user_resp.raise_for_status.return_value = None

    with (
        patch("it_agent.tools.tickets.ServiceNowClient") as mock_sn_cls,
        patch("it_agent.tools.tickets.AsyncWebClient", return_value=mock_slack),
    ):
        mock_sn = AsyncMock()
        mock_sn.create_incident.return_value = incident
        mock_sn._client.get.return_value = mock_sn_user_resp
        mock_sn_cls.return_value = mock_sn

        result = await create_ticket(
            title="VPN not working",
            description="Cannot connect to VPN",
            priority="medium",
            _settings=settings,
            _user_id="U123",
        )

    assert result["success"] is True
    assert result["channel_name"] == "inc0010001-gporcaro"
    assert result["channel_id"] == "C999"

    # Verify caller resolution looked up the user
    mock_slack.users_info.assert_any_call(user="U123")

    # Verify incident was created with resolved SN sys_id
    create_call_args = mock_sn.create_incident.call_args[0][0]
    assert create_call_args["caller_id"] == "sn_user_123"

    mock_slack.conversations_create.assert_called_once_with(
        name="inc0010001-gporcaro", is_private=True
    )
    mock_slack.conversations_invite.assert_called_once_with(channel="C999", users="U123")
    mock_slack.chat_postMessage.assert_called_once()
    posted_text = mock_slack.chat_postMessage.call_args.kwargs["text"]
    assert "INC0010001" in posted_text
    assert "View in ServiceNow" in posted_text


@pytest.mark.asyncio
async def test_create_ticket_slack_failure_still_returns_incident():
    """If Slack channel creation fails, the incident is still returned successfully."""
    settings = _fake_settings()
    incident = _normalized_incident()

    mock_slack = AsyncMock()
    mock_slack.users_info.side_effect = Exception("Slack API error")

    with (
        patch("it_agent.tools.tickets.ServiceNowClient") as mock_sn_cls,
        patch("it_agent.tools.tickets.AsyncWebClient", return_value=mock_slack),
    ):
        mock_sn = AsyncMock()
        mock_sn.create_incident.return_value = incident
        mock_sn_cls.return_value = mock_sn

        result = await create_ticket(
            title="VPN not working",
            description="Cannot connect to VPN",
            priority="medium",
            _settings=settings,
            _user_id="U123",
        )

    assert result["success"] is True
    assert result["ticket"]["ticket_id"] == "INC0010001"
    # No channel info when Slack fails
    assert "channel_name" not in result
    assert "channel_id" not in result
