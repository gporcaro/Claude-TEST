"""Real-time dashboard server for the IT bot."""

from __future__ import annotations

import asyncio
import hmac
import json
import secrets
import smtplib
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path

import aiosqlite
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
    db_path: str = str(Path(__file__).parent.parent / "interactions.db")


settings = DashboardSettings()
AUTHORIZED: set[str] = {e.strip().lower() for e in settings.authorized_emails.split(",") if e.strip()}

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_db: Optional[aiosqlite.Connection] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db
    _db = await aiosqlite.connect(settings.db_path)
    _db.row_factory = aiosqlite.Row
    # WAL must be set before query_only since changing journal mode is a write
    try:
        await _db.execute("PRAGMA journal_mode = WAL")
    except Exception:
        pass  # May fail if another process holds an exclusive lock
    await _db.execute("PRAGMA query_only = ON")
    yield
    if _db:
        await _db.close()
        _db = None

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="IT Bot Dashboard", lifespan=lifespan)

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

_SESSIONS_FILE = Path(__file__).parent / ".sessions.json"


def _load_sessions() -> None:
    """Load persisted sessions from disk, discarding expired ones."""
    if not _SESSIONS_FILE.exists():
        return
    try:
        data = json.loads(_SESSIONS_FILE.read_text())
        now = time.time()
        for token, s in data.items():
            if s["expires_at"] > now:
                _sessions[token] = SessionData(
                    email=s["email"],
                    created_at=s["created_at"],
                    expires_at=s["expires_at"],
                )
    except Exception:
        pass  # Corrupt file — start fresh


def _save_sessions() -> None:
    """Persist current sessions to disk."""
    data = {
        token: {"email": s.email, "created_at": s.created_at, "expires_at": s.expires_at}
        for token, s in _sessions.items()
    }
    _SESSIONS_FILE.write_text(json.dumps(data))


_load_sessions()

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
        _save_sessions()
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
    _save_sessions()

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
        _save_sessions()
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
# Live dashboard init (seed from DB on page load)
# ---------------------------------------------------------------------------

@app.get("/api/live/init")
async def live_init(_session: SessionData = Depends(require_auth)) -> JSONResponse:
    """Return active tickets, stats, and recent tool calls from DB to seed the live dashboard."""
    today_cutoff = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).isoformat()

    # Active (non-resolved/closed) tickets
    async with _db.execute(
        """SELECT i.id, i.ticket_id, i.channel_id, i.priority, i.status,
                  i.requester_name, i.category,
                  (SELECT m.content FROM interaction_messages m
                   WHERE m.interaction_id = i.id AND m.role = 'user'
                   ORDER BY m.id LIMIT 1) AS first_message
           FROM interactions i
           WHERE i.status NOT IN ('resolved', 'closed')
           ORDER BY i.created_at DESC"""
    ) as cur:
        tickets = [dict(r) for r in await cur.fetchall()]

    # Today's resolved count
    async with _db.execute(
        "SELECT COUNT(*) FROM interactions WHERE status = 'resolved' AND resolved_at >= ?",
        [today_cutoff],
    ) as cur:
        resolved_today = (await cur.fetchone())[0]

    # Total interactions
    async with _db.execute("SELECT COUNT(*) FROM interactions") as cur:
        total_interactions = (await cur.fetchone())[0]

    # Recent tool calls (last 10)
    async with _db.execute(
        """SELECT tool_name, args, result_summary, timestamp
           FROM interaction_tool_calls
           ORDER BY id DESC LIMIT 10"""
    ) as cur:
        tool_calls = [dict(r) for r in await cur.fetchall()]

    # Messages in the last hour (for msgs/hr stat)
    one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    async with _db.execute(
        "SELECT COUNT(*) FROM interaction_messages WHERE role = 'user' AND timestamp >= ?",
        [one_hour_ago],
    ) as cur:
        msgs_last_hour = (await cur.fetchone())[0]

    return JSONResponse({
        "tickets": tickets,
        "resolved_today": resolved_today,
        "total_interactions": total_interactions,
        "tool_calls": tool_calls,
        "msgs_last_hour": msgs_last_hour,
    })

# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _range_filter(range_param: str) -> Optional[str]:
    """Convert a range parameter to an ISO 8601 cutoff string."""
    now = datetime.now(timezone.utc)
    if range_param == "today":
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif range_param == "7d":
        cutoff = now - timedelta(days=7)
    elif range_param == "30d":
        cutoff = now - timedelta(days=30)
    else:
        return None
    return cutoff.isoformat()

# ---------------------------------------------------------------------------
# Reporting page route
# ---------------------------------------------------------------------------

@app.get("/reporting")
async def reporting_page(_session: SessionData = Depends(require_auth)) -> FileResponse:
    return FileResponse(STATIC_DIR / "reporting.html")

# ---------------------------------------------------------------------------
# Reporting API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/reporting/kpis")
async def reporting_kpis(
    range: str = "all",
    _session: SessionData = Depends(require_auth),
) -> JSONResponse:
    cutoff = _range_filter(range)
    where = "WHERE created_at >= ?" if cutoff else ""
    params: list = [cutoff] if cutoff else []

    async with _db.execute(f"SELECT COUNT(*) FROM interactions {where}", params) as cur:
        total = (await cur.fetchone())[0]

    async with _db.execute(
        f"SELECT COUNT(*) FROM interactions {where + ' AND' if where else 'WHERE'} resolved_by_bot = 1",
        params,
    ) as cur:
        bot_resolved = (await cur.fetchone())[0]

    async with _db.execute(
        f"SELECT COALESCE(AVG(message_count), 0) FROM interactions {where}",
        params,
    ) as cur:
        avg_messages = round((await cur.fetchone())[0], 1)

    today_cutoff = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    async with _db.execute(
        "SELECT COUNT(*) FROM interactions WHERE created_at >= ?",
        [today_cutoff],
    ) as cur:
        today_count = (await cur.fetchone())[0]

    bot_pct = round(bot_resolved / total * 100, 1) if total > 0 else 0

    return JSONResponse({
        "total": total,
        "bot_resolved_pct": bot_pct,
        "avg_messages": avg_messages,
        "today_count": today_count,
    })


@app.get("/api/reporting/interactions")
async def reporting_interactions(
    range: str = "all",
    status: str = "",
    source: str = "",
    category: str = "",
    sort: str = "created_at",
    order: str = "desc",
    page: int = 1,
    per_page: int = 50,
    _session: SessionData = Depends(require_auth),
) -> JSONResponse:
    conditions: list[str] = []
    params: list = []

    cutoff = _range_filter(range)
    if cutoff:
        conditions.append("created_at >= ?")
        params.append(cutoff)
    if status:
        conditions.append("status = ?")
        params.append(status)
    if source:
        conditions.append("source = ?")
        params.append(source)
    if category:
        conditions.append("category = ?")
        params.append(category)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    # Whitelist sort columns
    allowed_sorts = {"created_at", "status", "source", "category", "requester_name", "message_count"}
    sort_col = sort if sort in allowed_sorts else "created_at"
    sort_dir = "ASC" if order.lower() == "asc" else "DESC"

    # Total count
    async with _db.execute(f"SELECT COUNT(*) FROM interactions {where}", params) as cur:
        total = (await cur.fetchone())[0]

    offset = (max(page, 1) - 1) * per_page
    query = f"""
        SELECT id, ticket_id, source, requester_name, category, subcategory,
               status, resolved_by_bot, priority, message_count, tool_call_count,
               created_at, resolved_at
        FROM interactions {where}
        ORDER BY {sort_col} {sort_dir}
        LIMIT ? OFFSET ?
    """
    async with _db.execute(query, params + [per_page, offset]) as cur:
        rows = [dict(r) for r in await cur.fetchall()]

    return JSONResponse({
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max(1, -(-total // per_page)),
        "interactions": rows,
    })


@app.get("/api/reporting/breakdowns")
async def reporting_breakdowns(
    range: str = "all",
    _session: SessionData = Depends(require_auth),
) -> JSONResponse:
    cutoff = _range_filter(range)
    where = "WHERE created_at >= ?" if cutoff else ""
    params: list = [cutoff] if cutoff else []

    async def group_by(col: str) -> list[dict]:
        q = f"SELECT {col} AS label, COUNT(*) AS count FROM interactions {where} GROUP BY {col} ORDER BY count DESC"
        async with _db.execute(q, params) as cur:
            return [{"label": r["label"] or "unknown", "count": r["count"]} for r in await cur.fetchall()]

    by_category = await group_by("category")
    by_source = await group_by("source")
    by_status = await group_by("status")

    # Top requesters
    rq = f"""
        SELECT requester_name AS label, COUNT(*) AS count
        FROM interactions {where}
        GROUP BY requester_name ORDER BY count DESC LIMIT 10
    """
    async with _db.execute(rq, params) as cur:
        top_requesters = [{"label": r["label"] or "unknown", "count": r["count"]} for r in await cur.fetchall()]

    return JSONResponse({
        "by_category": by_category,
        "by_source": by_source,
        "by_status": by_status,
        "top_requesters": top_requesters,
    })


@app.get("/api/reporting/interaction/{interaction_id}")
async def reporting_interaction_detail(
    interaction_id: int,
    _session: SessionData = Depends(require_auth),
) -> JSONResponse:
    async with _db.execute(
        "SELECT * FROM interactions WHERE id = ?", [interaction_id]
    ) as cur:
        row = await cur.fetchone()
        if not row:
            return JSONResponse({"error": "Not found"}, status_code=404)
        interaction = dict(row)

    async with _db.execute(
        "SELECT role, content, timestamp FROM interaction_messages WHERE interaction_id = ? ORDER BY id",
        [interaction_id],
    ) as cur:
        messages = [dict(r) for r in await cur.fetchall()]

    async with _db.execute(
        "SELECT tool_name, args, result_summary, timestamp FROM interaction_tool_calls WHERE interaction_id = ? ORDER BY id",
        [interaction_id],
    ) as cur:
        tool_calls = [dict(r) for r in await cur.fetchall()]

    return JSONResponse({
        "interaction": interaction,
        "messages": messages,
        "tool_calls": tool_calls,
    })


@app.get("/api/reporting/interaction/by-ticket/{ticket_id}")
async def reporting_interaction_by_ticket(
    ticket_id: str,
    _session: SessionData = Depends(require_auth),
) -> JSONResponse:
    """Look up an interaction by its ticket_id and return full detail."""
    async with _db.execute(
        "SELECT id FROM interactions WHERE ticket_id = ?", [ticket_id]
    ) as cur:
        row = await cur.fetchone()
        if not row:
            return JSONResponse({"error": "Not found"}, status_code=404)
        interaction_id = row[0]
    # Delegate to the existing detail handler
    return await reporting_interaction_detail(interaction_id, _session)

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
