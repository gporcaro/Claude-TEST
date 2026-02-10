"""Ticket tool wrappers — bridge between agent executor and ServiceNow."""

from __future__ import annotations

import logging

from slack_sdk.web.async_client import AsyncWebClient

from it_agent.config import Settings
from it_agent.servicenow.client import _PRIORITY_TO_URGENCY, _STATUS_TO_STATE, ServiceNowClient

logger = logging.getLogger(__name__)


async def create_ticket(
    title: str,
    description: str,
    priority: str = "medium",
    category: str = "",
    _settings: Settings | None = None,
    _user_id: str = "unknown",
    **_,
) -> dict:
    """Create a new IT support ticket in ServiceNow."""
    if _settings is None:
        return {"error": "Settings not configured"}

    client = ServiceNowClient(
        _settings.sn_instance_url, _settings.sn_username, _settings.sn_password
    )
    try:
        incident = await client.create_incident(
            {
                "title": title,
                "description": description,
                "priority": priority,
                "category": category,
                "caller_id": _user_id,
            }
        )
    finally:
        await client.close()

    result: dict = {"success": True, "ticket": incident}

    # Auto-create a private Slack channel for the incident
    logger.debug(
        "Channel creation: _settings type=%s, _user_id=%r, ticket_id=%s",
        type(_settings).__name__,
        _user_id,
        incident.get("ticket_id"),
    )
    try:
        slack = AsyncWebClient(token=_settings.slack_bot_token)

        # Look up user's display name for channel naming
        user_info = await slack.users_info(user=_user_id)
        username = user_info["user"]["name"]

        # Slack channel names: lowercase, no spaces, max 80 chars
        inc_number = incident["ticket_id"].lower()
        channel_name = f"{inc_number}-{username}"[:80]

        channel_resp = await slack.conversations_create(
            name=channel_name, is_private=True
        )
        channel_id = channel_resp["channel"]["id"]

        # Invite the requester to the channel
        await slack.conversations_invite(channel=channel_id, users=_user_id)

        # Post initial message with incident summary + ServiceNow link
        sn_link = (
            f"{_settings.sn_instance_url}/incident.do"
            f"?sysparm_query=number={incident['ticket_id']}"
        )
        message = (
            f"*Incident {incident['ticket_id']}*\n"
            f"*Title:* {incident['title']}\n"
            f"*Priority:* {incident['priority']}\n"
            f"*Description:* {incident['description']}\n\n"
            f"<{sn_link}|View in ServiceNow>"
        )
        await slack.chat_postMessage(channel=channel_id, text=message)

        result["channel_name"] = channel_name
        result["channel_id"] = channel_id
    except Exception:
        logger.warning(
            "Failed to create Slack channel for %s",
            incident["ticket_id"],
            exc_info=True,
        )

    return result


async def get_ticket(
    ticket_id: str,
    _settings: Settings | None = None,
    **_,
) -> dict:
    """Retrieve a ticket by its incident number (e.g. INC0010001)."""
    if _settings is None:
        return {"error": "Settings not configured"}

    client = ServiceNowClient(
        _settings.sn_instance_url, _settings.sn_username, _settings.sn_password
    )
    try:
        incident = await client.get_incident(ticket_id)
        if incident is None:
            return {"error": f"Ticket {ticket_id} not found"}
        return {"ticket": incident}
    finally:
        await client.close()


async def update_ticket(
    ticket_id: str,
    status: str | None = None,
    priority: str | None = None,
    assignee_id: str | None = None,
    comment: str | None = None,
    close_notes: str | None = None,
    _settings: Settings | None = None,
    _user_id: str = "unknown",
    **_,
) -> dict:
    """Update ticket fields and/or add a comment."""
    if _settings is None:
        return {"error": "Settings not configured"}

    client = ServiceNowClient(
        _settings.sn_instance_url, _settings.sn_username, _settings.sn_password
    )
    try:
        # First look up the incident to get sys_id
        incident = await client.get_incident(ticket_id)
        if incident is None:
            return {"error": f"Ticket {ticket_id} not found"}

        update_data: dict = {}
        if status:
            update_data["status"] = status
        if priority:
            update_data["priority"] = priority
        if assignee_id:
            update_data["assignee_id"] = assignee_id
        if comment:
            update_data["comment"] = comment
        if close_notes:
            update_data["close_notes"] = close_notes

        updated = await client.update_incident(
            incident["sys_id"], update_data, current_state=incident.get("_raw_state", "1")
        )
        return {"success": True, "ticket": updated}
    finally:
        await client.close()


async def list_tickets(
    status: str | None = None,
    priority: str | None = None,
    requester_id: str | None = None,
    limit: int = 10,
    _settings: Settings | None = None,
    **_,
) -> dict:
    """List tickets with optional filters."""
    if _settings is None:
        return {"error": "Settings not configured"}

    # Build ServiceNow encoded query
    parts: list[str] = []
    if status:
        state = _STATUS_TO_STATE.get(status, status)
        parts.append(f"state={state}")
    if priority:
        urgency = _PRIORITY_TO_URGENCY.get(priority, priority)
        parts.append(f"urgency={urgency}")
    if requester_id:
        parts.append(f"caller_id={requester_id}")
    query = "^".join(parts)

    client = ServiceNowClient(
        _settings.sn_instance_url, _settings.sn_username, _settings.sn_password
    )
    try:
        incidents = await client.list_incidents(query=query, limit=limit)
        return {"tickets": incidents, "count": len(incidents)}
    finally:
        await client.close()
