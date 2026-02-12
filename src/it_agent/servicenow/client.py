"""Async ServiceNow REST API client."""

from __future__ import annotations

import httpx

# --- Field mappings ---

_PRIORITY_TO_URGENCY: dict[str, str] = {
    "low": "3",
    "medium": "2",
    "high": "1",
    "critical": "1",
}

_URGENCY_TO_PRIORITY: dict[str, str] = {
    "1": "high",
    "2": "medium",
    "3": "low",
}

_STATUS_TO_STATE: dict[str, str] = {
    "open": "1",
    "in_progress": "2",
    "waiting": "3",
    "resolved": "6",
    "closed": "7",
}

_STATE_TO_STATUS: dict[str, str] = {
    "1": "open",
    "2": "in_progress",
    "3": "waiting",
    "6": "resolved",
    "7": "closed",
}


def _normalize_incident(record: dict) -> dict:
    """Convert a raw ServiceNow incident record to bot-friendly fields."""
    return {
        "ticket_id": record.get("number", ""),
        "sys_id": record.get("sys_id", ""),
        "title": record.get("short_description", ""),
        "description": record.get("description", ""),
        "priority": _URGENCY_TO_PRIORITY.get(str(record.get("urgency", "3")), "medium"),
        "status": _STATE_TO_STATUS.get(str(record.get("state", "1")), "open"),
        "category": record.get("category", ""),
        "subcategory": record.get("subcategory", ""),
        "contact_type": record.get("contact_type", ""),
        "requester_id": record.get("caller_id", ""),
        "assignee_id": record.get("assigned_to", ""),
        "additional_assignee_list": record.get("additional_assignee_list", ""),
        "created_at": record.get("sys_created_on", ""),
        "updated_at": record.get("sys_updated_on", ""),
    }


def _normalize_kb_article(record: dict) -> dict:
    """Convert a raw ServiceNow KB article to bot-friendly fields."""
    return {
        "id": record.get("number", record.get("sys_id", "")),
        "title": record.get("short_description", ""),
        "content": record.get("text", ""),
        "category": record.get("kb_category", ""),
    }


class ServiceNowClient:
    """Async client for ServiceNow Table API."""

    def __init__(self, instance_url: str, username: str, password: str) -> None:
        self.base_url = f"{instance_url.rstrip('/')}/api/now"
        self._client = httpx.AsyncClient(
            auth=(username, password),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=30.0,
        )

    # --- Incident CRUD ---

    async def create_incident(self, data: dict) -> dict:
        """Create a new incident. *data* uses bot-friendly field names."""
        payload: dict = {
            "short_description": data.get("title", ""),
            "description": data.get("description", ""),
        }
        priority = data.get("priority", "medium")
        payload["urgency"] = _PRIORITY_TO_URGENCY.get(priority, "2")
        if priority == "critical":
            payload["impact"] = "1"

        if data.get("category"):
            payload["category"] = data["category"]
        if data.get("caller_id"):
            payload["caller_id"] = data["caller_id"]
        if data.get("assigned_to"):
            payload["assigned_to"] = data["assigned_to"]

        resp = await self._client.post(
            f"{self.base_url}/table/incident", json=payload
        )
        resp.raise_for_status()
        return _normalize_incident(resp.json()["result"])

    async def get_incident(self, number: str) -> dict | None:
        """Fetch an incident by its number (e.g. INC0010001)."""
        resp = await self._client.get(
            f"{self.base_url}/table/incident",
            params={
                "sysparm_query": f"number={number}",
                "sysparm_limit": "1",
            },
        )
        resp.raise_for_status()
        results = resp.json()["result"]
        if not results:
            return None
        normalized = _normalize_incident(results[0])
        # Preserve raw state for transition logic
        normalized["_raw_state"] = str(results[0].get("state", "1"))
        return normalized

    async def update_incident(
        self, sys_id: str, data: dict, current_state: str = "1"
    ) -> dict:
        """Update an incident by sys_id. *data* uses bot-friendly field names.

        *current_state* is the raw SN state value so we can step through
        required intermediate transitions (e.g. New→In Progress→Resolved).
        """
        target_status = data.get("status")
        target_state = _STATUS_TO_STATE.get(target_status, "") if target_status else ""

        # ServiceNow enforces state-machine rules. Walk through intermediate
        # states when jumping from New (1) to Resolved (6) or Closed (7).
        transition_order = ["1", "2", "6", "7"]

        if target_state and target_state in transition_order:
            cur_idx = (
                transition_order.index(current_state)
                if current_state in transition_order
                else 0
            )
            tgt_idx = transition_order.index(target_state)

            # Step through each intermediate state before the final one
            for step_state in transition_order[cur_idx + 1 : tgt_idx]:
                await self._client.patch(
                    f"{self.base_url}/table/incident/{sys_id}",
                    json={"state": step_state},
                )

        # Build final payload
        payload: dict = {}
        if target_state:
            payload["state"] = target_state
        if "priority" in data:
            payload["urgency"] = _PRIORITY_TO_URGENCY.get(data["priority"], "2")
        if "assignee_id" in data:
            payload["assigned_to"] = data["assignee_id"]
        if "requester_id" in data:
            payload["caller_id"] = data["requester_id"]
        if "comment" in data:
            payload["comments"] = data["comment"]
        if "additional_assignee_list" in data:
            payload["additional_assignee_list"] = data["additional_assignee_list"]
        if "description" in data:
            payload["description"] = data["description"]

        # ServiceNow requires close_code + close_notes for resolved/closed
        if target_status in ("resolved", "closed"):
            close_notes = data.get("close_notes") or data.get("comment") or "Resolved"
            payload["close_code"] = data.get("close_code", "Solved (Permanently)")
            payload["close_notes"] = close_notes

        resp = await self._client.patch(
            f"{self.base_url}/table/incident/{sys_id}", json=payload
        )
        resp.raise_for_status()
        return _normalize_incident(resp.json()["result"])

    async def list_incidents(
        self, query: str = "", limit: int = 10
    ) -> list[dict]:
        """List incidents with an optional encoded query."""
        params: dict = {"sysparm_limit": str(limit)}
        if query:
            params["sysparm_query"] = query
        resp = await self._client.get(
            f"{self.base_url}/table/incident", params=params
        )
        resp.raise_for_status()
        return [_normalize_incident(r) for r in resp.json()["result"]]

    # --- Knowledge Base ---

    async def list_kb_articles(self, limit: int = 500) -> list[dict]:
        """Fetch all published KB articles for indexing (no keyword filter)."""
        encoded_query = (
            "workflow_state=published"
            "^retiredISEMPTY"
            "^kb_category!=NULL"
        )
        resp = await self._client.get(
            f"{self.base_url}/table/kb_knowledge",
            params={
                "sysparm_query": encoded_query,
                "sysparm_limit": str(limit),
                "sysparm_fields": "number,short_description,text,sys_id,kb_category",
            },
        )
        resp.raise_for_status()
        return [_normalize_kb_article(r) for r in resp.json()["result"]]

    async def search_kb_articles(
        self, query: str, limit: int = 5
    ) -> list[dict]:
        """Search published KB articles by keyword.

        Splits the query into individual keywords and builds a ServiceNow
        query that matches ANY keyword in either the title or body, which
        is far more effective than matching the entire phrase literally.
        """
        # Stop words that add noise to LIKE searches
        _stop = {"a", "an", "the", "to", "how", "do", "i", "is", "in", "on", "for", "my", "me"}
        keywords = [w for w in query.split() if w.lower() not in _stop and len(w) > 1]
        if not keywords:
            keywords = query.split()[:3]

        # Build OR conditions: each keyword can match in title OR body
        keyword_conditions: list[str] = []
        for kw in keywords:
            keyword_conditions.append(f"short_descriptionLIKE{kw}")
            keyword_conditions.append(f"textLIKE{kw}")

        encoded_query = (
            "workflow_state=published"
            "^retiredISEMPTY"
            "^short_descriptionNOT LIKEtest"
            "^kb_category!=NULL"
            "^" + "^OR".join(keyword_conditions)
        )
        resp = await self._client.get(
            f"{self.base_url}/table/kb_knowledge",
            params={
                "sysparm_query": encoded_query,
                "sysparm_limit": str(limit),
                "sysparm_fields": "number,short_description,text,sys_id,kb_category",
            },
        )
        resp.raise_for_status()
        return [_normalize_kb_article(r) for r in resp.json()["result"]]

    # --- Lifecycle ---

    async def close(self) -> None:
        await self._client.aclose()
