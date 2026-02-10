"""Real-time dashboard server for the IT bot."""

from __future__ import annotations

import asyncio
import hmac
import json
import secrets
import smtplib
import time
from collections import deque
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

from fastapi import Cookie, Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class DashboardSettings(BaseSettings):
    model_config = {"env_file": str(Path(__file__).parent.parent / ".env"), "env_file_encoding": "utf-8", "extra": "ignore"}
    smtp_email: str = ""
    smtp_app_password: str = ""
    authorized_emails: str = "geporcaro@gmail.com,juan.pablo.buffone@gmail.com"
    session_lifetime_hours: int = 8
    mfa_code_expiry_seconds: int = 300          # 5 min
    mfa_rate_limit_max: int = 3                 # 3 codes per window
    mfa_rate_limit_window_seconds: int = 600    # 10 min


settings = DashboardSettings()
AUTHORIZED: set[str] = {e.strip().lower() for e in settings.authorized_emails.split(",") if e.strip()}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="IT Bot Dashboard")

# In-memory ring buffer of the last 200 events
_events: deque[dict] = deque(maxlen=200)

# Connected WebSocket clients
_clients: set[WebSocket] = set()

STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# In-memory auth stores
# ---------------------------------------------------------------------------

@dataclass
class SessionData:
    email: str
    created_at: float
    expires_at: float


@dataclass
class PendingCode:
    code: str
    email: str
    expires_at: float
    attempts: int = 0


_sessions: dict[str, SessionData] = {}          # token → session
_pending_codes: dict[str, PendingCode] = {}     # email → pending code
_code_request_log: dict[str, list[float]] = {}  # email → timestamps

# ---------------------------------------------------------------------------
# SMTP helper
# ---------------------------------------------------------------------------

def _send_code_email(to_email: str, code: str) -> None:
    """Send the MFA code via Gmail SMTP (blocking — run in thread)."""
    msg = EmailMessage()
    msg["Subject"] = f"Your dashboard login code: {code}"
    msg["From"] = settings.smtp_email
    msg["To"] = to_email
    msg.set_content(
        f"Your one-time login code is:\n\n    {code}\n\n"
        "This code expires in 5 minutes. If you did not request this, ignore this email."
    )
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(settings.smtp_email, settings.smtp_app_password)
        smtp.send_message(msg)

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

class AuthRequired(Exception):
    """Raised when a request lacks a valid session."""


@app.exception_handler(AuthRequired)
async def _auth_required_handler(request: Request, exc: AuthRequired) -> Response:
    return RedirectResponse(url="/login", status_code=302)


def _valid_session(token: Optional[str]) -> Optional[SessionData]:
    if not token:
        return None
    session = _sessions.get(token)
    if session is None:
        return None
    if time.time() > session.expires_at:
        _sessions.pop(token, None)
        return None
    return session


async def require_auth(session_token: Optional[str] = Cookie(default=None)) -> SessionData:
    session = _valid_session(session_token)
    if session is None:
        raise AuthRequired()
    return session

# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

class CodeRequest(BaseModel):
    email: str


class CodeVerify(BaseModel):
    email: str
    code: str


@app.get("/login")
async def login_page() -> FileResponse:
    """Serve the login page."""
    return FileResponse(STATIC_DIR / "login.html")


@app.post("/auth/request-code")
async def request_code(body: CodeRequest) -> JSONResponse:
    """Validate email, rate-limit, generate code, send via SMTP."""
    email = body.email.strip().lower()

    # Always return same generic response to avoid email enumeration
    ok_response = JSONResponse({"ok": True, "message": "If that email is authorized, a code has been sent."})

    if email not in AUTHORIZED:
        return ok_response

    # Rate limiting
    now = time.time()
    window_start = now - settings.mfa_rate_limit_window_seconds
    log = _code_request_log.setdefault(email, [])
    # Prune old entries
    log[:] = [ts for ts in log if ts > window_start]
    if len(log) >= settings.mfa_rate_limit_max:
        return JSONResponse(
            {"ok": False, "message": "Too many code requests. Please wait a few minutes."},
            status_code=429,
        )
    log.append(now)

    # Generate and store code
    code = f"{secrets.randbelow(10**6):06d}"
    _pending_codes[email] = PendingCode(
        code=code,
        email=email,
        expires_at=now + settings.mfa_code_expiry_seconds,
    )

    # Send email in background thread
    try:
        await asyncio.to_thread(_send_code_email, email, code)
    except Exception:
        # Don't leak SMTP errors to client
        pass

    return ok_response


@app.post("/auth/verify-code")
async def verify_code(body: CodeVerify) -> JSONResponse:
    """Verify the MFA code and create a session."""
    email = body.email.strip().lower()
    pending = _pending_codes.get(email)

    if pending is None:
        return JSONResponse({"ok": False, "message": "No pending code. Please request a new one."}, status_code=400)

    # Expired
    if time.time() > pending.expires_at:
        _pending_codes.pop(email, None)
        return JSONResponse({"ok": False, "message": "Code expired. Please request a new one."}, status_code=400)

    # Attempt limit
    pending.attempts += 1
    if pending.attempts > 5:
        _pending_codes.pop(email, None)
        return JSONResponse({"ok": False, "message": "Too many attempts. Please request a new code."}, status_code=400)

    # Constant-time comparison
    if not hmac.compare_digest(body.code.strip(), pending.code):
        remaining = 5 - pending.attempts
        return JSONResponse({"ok": False, "message": f"Invalid code. {remaining} attempt(s) remaining."}, status_code=401)

    # Success — clean up pending code
    _pending_codes.pop(email, None)

    # Create session
    token = secrets.token_urlsafe(32)
    now = time.time()
    _sessions[token] = SessionData(
        email=email,
        created_at=now,
        expires_at=now + settings.session_lifetime_hours * 3600,
    )

    response = JSONResponse({"ok": True})
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=settings.session_lifetime_hours * 3600,
        path="/",
    )
    return response


@app.post("/auth/logout")
async def logout(session_token: Optional[str] = Cookie(default=None)) -> JSONResponse:
    """Clear session and delete cookie."""
    if session_token:
        _sessions.pop(session_token, None)
    response = JSONResponse({"ok": True})
    response.delete_cookie(key="session_token", path="/")
    return response

# ---------------------------------------------------------------------------
# Event ingestion (unauthenticated — internal bot traffic)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Protected routes
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """WebSocket endpoint for browser connections.

    Sends recent event history on connect, then streams new events.
    """
    # Check auth before accepting
    token = ws.cookies.get("session_token")
    if not _valid_session(token):
        await ws.close(code=1008, reason="Authentication required")
        return

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
async def index(_session: SessionData = Depends(require_auth)) -> FileResponse:
    """Serve the dashboard HTML."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/static/{path:path}")
async def static_file(path: str, _session: SessionData = Depends(require_auth)) -> FileResponse:
    """Serve static files with auth and path traversal protection."""
    resolved = (STATIC_DIR / path).resolve()
    if not str(resolved).startswith(str(STATIC_DIR.resolve())):
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    if not resolved.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(resolved)

# ---------------------------------------------------------------------------
# Broadcast
# ---------------------------------------------------------------------------

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
