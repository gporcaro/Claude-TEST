"""User lookup tool — resolve names to Slack and ServiceNow identities."""

from __future__ import annotations

import logging

from slack_sdk.web.async_client import AsyncWebClient

from it_agent.config import Settings
from it_agent.servicenow.client import ServiceNowClient

logger = logging.getLogger(__name__)


async def lookup_user(
    name: str,
    _settings: Settings | None = None,
    **_,
) -> dict:
    """Search for a user by name across Slack and ServiceNow.

    Returns matching candidates with Slack user ID, ServiceNow sys_id,
    display name, and email so the agent can confirm with the requester.
    """
    if _settings is None:
        return {"error": "Settings not configured"}

    # 1. Search Slack users by name
    slack = AsyncWebClient(token=_settings.slack_bot_token)
    candidates: list[dict] = []
    search_lower = name.lower()

    try:
        cursor = None
        while True:
            resp = await slack.users_list(limit=200, cursor=cursor)
            for member in resp.get("members", []):
                if member.get("deleted") or member.get("is_bot"):
                    continue
                profile = member.get("profile", {})
                real_name = profile.get("real_name", "")
                display_name = profile.get("display_name", "")
                first_name = profile.get("first_name", "")
                last_name = profile.get("last_name", "")

                # Match against real name, display name, or first/last name
                searchable = f"{real_name} {display_name} {first_name} {last_name}".lower()
                if search_lower in searchable:
                    candidates.append({
                        "slack_id": member["id"],
                        "real_name": real_name,
                        "display_name": display_name,
                        "email": profile.get("email", ""),
                    })

            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
    except Exception:
        logger.warning("Slack user search failed", exc_info=True)
        return {"error": "Failed to search Slack users"}

    # 2. Look up ServiceNow sys_id for each candidate via email, then by name.
    #    If no Slack candidates, search ServiceNow directly by name.
    sn_client = ServiceNowClient(
        _settings.sn_instance_url, _settings.sn_username, _settings.sn_password
    )
    try:
        # If no Slack matches, search ServiceNow directly by name
        if not candidates:
            try:
                resp = await sn_client._client.get(
                    f"{sn_client.base_url}/table/sys_user",
                    params={
                        "sysparm_query": f"nameLIKE{name}^active=true",
                        "sysparm_limit": "5",
                        "sysparm_fields": "sys_id,name,email",
                    },
                )
                resp.raise_for_status()
                for sn_user in resp.json().get("result", []):
                    candidates.append({
                        "slack_id": None,
                        "real_name": sn_user.get("name", ""),
                        "display_name": "",
                        "email": sn_user.get("email", ""),
                        "servicenow_sys_id": sn_user["sys_id"],
                        "source": "servicenow",
                    })
            except Exception:
                logger.debug("SN direct name lookup failed for %s", name)

            if not candidates:
                return {
                    "candidates": [],
                    "message": f"No users found matching '{name}' in Slack or ServiceNow.",
                }
        else:
            # Enrich Slack candidates with ServiceNow sys_id
            for candidate in candidates:
                candidate["servicenow_sys_id"] = None

                # Try email first
                email = candidate.get("email", "")
                if email:
                    try:
                        resp = await sn_client._client.get(
                            f"{sn_client.base_url}/table/sys_user",
                            params={
                                "sysparm_query": f"email={email}",
                                "sysparm_limit": "1",
                                "sysparm_fields": "sys_id,name,email",
                            },
                        )
                        resp.raise_for_status()
                        results = resp.json().get("result", [])
                        if results:
                            candidate["servicenow_sys_id"] = results[0]["sys_id"]
                    except Exception:
                        logger.debug("SN email lookup failed for %s", email)

                # Fallback: search by name if email didn't match
                if not candidate["servicenow_sys_id"]:
                    real_name = candidate.get("real_name", "")
                    if real_name:
                        try:
                            resp = await sn_client._client.get(
                                f"{sn_client.base_url}/table/sys_user",
                                params={
                                    "sysparm_query": f"nameLIKE{real_name}",
                                    "sysparm_limit": "1",
                                    "sysparm_fields": "sys_id,name,email",
                                },
                            )
                            resp.raise_for_status()
                            results = resp.json().get("result", [])
                            if results:
                                candidate["servicenow_sys_id"] = results[0]["sys_id"]
                                candidate["servicenow_name"] = results[0].get("name", "")
                        except Exception:
                            logger.debug("SN name lookup failed for %s", real_name)
    finally:
        await sn_client.close()

    return {
        "candidates": candidates,
        "count": len(candidates),
        "message": (
            f"Found {len(candidates)} user(s) matching '{name}'."
            if len(candidates) != 1
            else f"Found user: {candidates[0]['real_name']} ({candidates[0]['email']})"
        ),
    }
