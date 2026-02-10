"""Fire-and-forget event emitter for the real-time dashboard."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_dashboard_url: str | None = None
_client: httpx.AsyncClient | None = None


def configure(dashboard_url: str) -> None:
    """Set the dashboard endpoint URL."""
    global _dashboard_url, _client
    _dashboard_url = dashboard_url.rstrip("/")
    _client = httpx.AsyncClient(timeout=2.0)


async def emit(event_type: str, data: dict[str, Any] | None = None) -> None:
    """Fire-and-forget event to the dashboard. Never raises."""
    if not _dashboard_url or not _client:
        return
    payload = {"type": event_type, "timestamp": time.time(), "data": data or {}}
    try:
        asyncio.create_task(_client.post(f"{_dashboard_url}/events", json=payload))
    except Exception:
        pass  # dashboard down — ignore
