"""Real-time dashboard server for the IT bot."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any

app = FastAPI(title="IT Bot Dashboard")

# In-memory ring buffer of the last 200 events
_events: deque[dict] = deque(maxlen=200)

# Connected WebSocket clients
_clients: set[WebSocket] = set()

STATIC_DIR = Path(__file__).parent / "static"


class Event(BaseModel):
    type: str
    timestamp: float
    data: dict[str, Any] = {}


@app.post("/events")
async def receive_event(event: Event) -> dict:
    """Receive an event from the bot and broadcast to all WebSocket clients."""
    payload = event.model_dump()
    _events.append(payload)
    await _broadcast(payload)
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """WebSocket endpoint for browser connections.

    Sends recent event history on connect, then streams new events.
    """
    await ws.accept()
    _clients.add(ws)
    try:
        # Send recent history
        for evt in _events:
            await ws.send_text(json.dumps(evt))
        # Keep connection alive — wait for client messages (pings / close)
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)


@app.get("/")
async def index() -> FileResponse:
    """Serve the dashboard HTML."""
    return FileResponse(STATIC_DIR / "index.html")


# Mount static files for any additional assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


async def _broadcast(payload: dict) -> None:
    """Send an event to all connected WebSocket clients."""
    if not _clients:
        return
    message = json.dumps(payload)
    stale: list[WebSocket] = []
    for ws in _clients:
        try:
            await ws.send_text(message)
        except Exception:
            stale.append(ws)
    for ws in stale:
        _clients.discard(ws)
