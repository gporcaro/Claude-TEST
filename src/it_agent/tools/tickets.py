"""Ticket tool wrappers — bridge between agent executor and ServiceNow."""

from __future__ import annotations

import logging

from slack_sdk.web.async_client import AsyncWebClient

from it_agent.config import Settings
from it_agent.servicenow.client import _PRIORITY_TO_URGENCY, _STATUS_TO_STATE, ServiceNowClient

logger = logging.getLogger(__name__)


async def create_incident_channel(
    incident: dict,
    _settings: Settings,
    _user_id: str,
) -> dict:
    """Create a private Slack channel for an incident and post the summary.

    Returns ``{channel_name, channel_id, summary_ts}`` on success, or ``{}``
    on failure.
    """
    try:
        slack = AsyncWebClient(token=_settings.slack_bot_token)

        # Look up user's display name for channel naming
        username = "support"
        if _user_id and _user_id != "unknown":
            try:
                user_info = await slack.users_info(user=_user_id)
                username = user_info["user"]["name"]
            except Exception:
                logger.debug("Could not resolve user %s for channel naming", _user_id)

        # Slack channel names: lowercase, no spaces, max 80 chars
        inc_number = incident["ticket_id"].lower()
        channel_name = f"{inc_number}-{username}"[:80]

        channel_resp = await slack.conversations_create(
            name=channel_name, is_private=True
        )
        channel_id = channel_resp["channel"]["id"]

        # Invite the requester to the channel (isolated so a failure here
        # does not prevent the summary message from being posted).
        if _user_id and _user_id != "unknown":
            try:
                await slack.conversations_invite(channel=channel_id, users=_user_id)
            except Exception as exc:
                if "already_in_channel" in str(exc):
                    logger.debug("Requester %s already in channel %s", _user_id, channel_id)
                else:
                    logger.warning(
                        "Failed to invite requester %s to channel %s: %s",
                        _user_id, channel_id, exc,
                    )

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
        msg_resp = await slack.chat_postMessage(channel=channel_id, text=message)

        return {
            "channel_name": channel_name,
            "channel_id": channel_id,
            "summary_ts": msg_resp["ts"],
        }
    except Exception:
        logger.warning(
            "Failed to create Slack channel for %s",
            incident.get("ticket_id", "unknown"),
            exc_info=True,
        )
        return {}


async def create_ticket(
    title: str,
    description: str,
    priority: str = "medium",
    category: str = "",
    create_channel: bool = True,
    _settings: Settings | None = None,
    _user_id: str = "unknown",
    **_,
) -> dict:
    """Create a new IT support ticket in ServiceNow."""
    if _settings is None:
        return {"error": "Settings not configured"}

    # Resolve Slack user ID → ServiceNow sys_id for the caller field.
    caller_id = ""
    if _user_id and _user_id != "unknown":
        try:
            slack = AsyncWebClient(token=_settings.slack_bot_token)
            user_info = await slack.users_info(user=_user_id)
            profile = user_info.get("user", {}).get("profile", {})
            email = profile.get("email", "")
            real_name = profile.get("real_name", "")

            sn_client = ServiceNowClient(
                _settings.sn_instance_url, _settings.sn_username, _settings.sn_password
            )
            try:
                # 1) Try email lookup first
                if email:
                    resp = await sn_client._client.get(
                        f"{sn_client.base_url}/table/sys_user",
                        params={
                            "sysparm_query": f"email={email}",
                            "sysparm_limit": "1",
                            "sysparm_fields": "sys_id",
                        },
                    )
                    resp.raise_for_status()
                    results = resp.json().get("result", [])
                    if results:
                        caller_id = results[0]["sys_id"]
                        logger.info("Resolved caller %s via email (%s) → SN sys_id %s", _user_id, email, caller_id)

                # 2) Fall back to name lookup if email didn't match
                if not caller_id and real_name:
                    resp = await sn_client._client.get(
                        f"{sn_client.base_url}/table/sys_user",
                        params={
                            "sysparm_query": f"name={real_name}",
                            "sysparm_limit": "1",
                            "sysparm_fields": "sys_id",
                        },
                    )
                    resp.raise_for_status()
                    results = resp.json().get("result", [])
                    if results:
                        caller_id = results[0]["sys_id"]
                        logger.info("Resolved caller %s via name (%s) → SN sys_id %s", _user_id, real_name, caller_id)

                if not caller_id:
                    logger.warning(
                        "Could not find SN user for Slack user %s (email=%s, name=%s)",
                        _user_id, email, real_name,
                    )
            finally:
                await sn_client.close()
        except Exception:
            logger.warning("Could not resolve Slack user %s to SN sys_id", _user_id, exc_info=True)

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
                "caller_id": caller_id,
            }
        )
    finally:
        await client.close()

    result: dict = {"success": True, "ticket": incident}

    # Optionally create a private Slack channel for the incident.
    # When create_channel=False (e.g. #help-it deferred flow), skip this
    # entirely and return ticket-only data.
    if create_channel:
        logger.debug(
            "Channel creation: _settings type=%s, _user_id=%r, ticket_id=%s",
            type(_settings).__name__,
            _user_id,
            incident.get("ticket_id"),
        )
        ch = await create_incident_channel(incident, _settings, _user_id)
        if ch:
            result["channel_name"] = ch["channel_name"]
            result["channel_id"] = ch["channel_id"]
            result["summary_ts"] = ch["summary_ts"]

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
    requester_id: str | None = None,
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
        if requester_id:
            update_data["requester_id"] = requester_id
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
