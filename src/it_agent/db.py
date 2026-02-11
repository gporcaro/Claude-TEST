"""Async SQLite database for recording bot interactions."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import aiosqlite

logger = logging.getLogger(__name__)

_db: Optional[aiosqlite.Connection] = None

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT,
    channel_id TEXT NOT NULL,
    thread_ts TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL,

    requester_slack_id TEXT NOT NULL,
    requester_name TEXT NOT NULL DEFAULT '',
    requester_email TEXT NOT NULL DEFAULT '',

    assignee_slack_id TEXT,
    assignee_name TEXT,

    category TEXT NOT NULL DEFAULT '',
    subcategory TEXT NOT NULL DEFAULT '',
    contact_type TEXT NOT NULL DEFAULT 'slack',
    priority TEXT NOT NULL DEFAULT 'medium',

    status TEXT NOT NULL DEFAULT 'open',
    resolved_by_bot INTEGER NOT NULL DEFAULT 0,
    close_notes TEXT,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_at TEXT,

    message_count INTEGER NOT NULL DEFAULT 0,
    tool_call_count INTEGER NOT NULL DEFAULT 0,
    first_response_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_interactions_requester ON interactions(requester_slack_id);
CREATE INDEX IF NOT EXISTS idx_interactions_assignee ON interactions(assignee_slack_id);
CREATE INDEX IF NOT EXISTS idx_interactions_category ON interactions(category);
CREATE INDEX IF NOT EXISTS idx_interactions_subcategory ON interactions(subcategory);
CREATE INDEX IF NOT EXISTS idx_interactions_status ON interactions(status);
CREATE INDEX IF NOT EXISTS idx_interactions_ticket ON interactions(ticket_id);
CREATE INDEX IF NOT EXISTS idx_interactions_created ON interactions(created_at);

CREATE TABLE IF NOT EXISTS interaction_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (interaction_id) REFERENCES interactions(id)
);

CREATE TABLE IF NOT EXISTS interaction_tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    args TEXT NOT NULL DEFAULT '{}',
    result_summary TEXT NOT NULL DEFAULT '',
    timestamp TEXT NOT NULL,
    FOREIGN KEY (interaction_id) REFERENCES interactions(id)
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_interaction ON interaction_tool_calls(interaction_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_name ON interaction_tool_calls(tool_name);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def init_db(db_path: str) -> None:
    """Open the database connection and create tables if needed."""
    global _db
    _db = await aiosqlite.connect(db_path)
    _db.row_factory = aiosqlite.Row
    await _db.executescript(_SCHEMA)
    await _db.commit()
    logger.info("Interactions database initialized at %s", db_path)


async def close_db() -> None:
    """Close the database connection."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None


async def create_interaction(
    *,
    channel_id: str,
    thread_ts: str = "",
    source: str,
    requester_slack_id: str,
    requester_name: str = "",
    requester_email: str = "",
    ticket_id: Optional[str] = None,
    category: str = "",
    subcategory: str = "",
    contact_type: str = "slack",
    priority: str = "medium",
) -> int:
    """Insert a new interaction record. Returns the interaction id."""
    if _db is None:
        raise RuntimeError("Database not initialized")
    now = _now_iso()
    cursor = await _db.execute(
        """INSERT INTO interactions
           (ticket_id, channel_id, thread_ts, source,
            requester_slack_id, requester_name, requester_email,
            category, subcategory, contact_type, priority,
            created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ticket_id, channel_id, thread_ts, source,
            requester_slack_id, requester_name, requester_email,
            category, subcategory, contact_type, priority,
            now, now,
        ),
    )
    await _db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def update_interaction(interaction_id: int, **fields: Any) -> None:
    """Update arbitrary fields on an interaction record."""
    if _db is None or not fields:
        return
    fields["updated_at"] = _now_iso()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [interaction_id]
    await _db.execute(
        f"UPDATE interactions SET {set_clause} WHERE id = ?",  # noqa: S608
        values,
    )
    await _db.commit()


async def increment_counts(
    interaction_id: int, messages: int = 0, tool_calls: int = 0,
) -> None:
    """Atomically increment message_count and/or tool_call_count."""
    if _db is None:
        return
    if messages:
        await _db.execute(
            "UPDATE interactions SET message_count = message_count + ?, updated_at = ? WHERE id = ?",
            (messages, _now_iso(), interaction_id),
        )
    if tool_calls:
        await _db.execute(
            "UPDATE interactions SET tool_call_count = tool_call_count + ?, updated_at = ? WHERE id = ?",
            (tool_calls, _now_iso(), interaction_id),
        )
    if messages or tool_calls:
        await _db.commit()


async def add_message(
    interaction_id: int, role: str, content: str,
) -> None:
    """Record an individual message within an interaction."""
    if _db is None:
        return
    await _db.execute(
        "INSERT INTO interaction_messages (interaction_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (interaction_id, role, content, _now_iso()),
    )
    await _db.commit()


async def add_tool_call(
    interaction_id: int,
    tool_name: str,
    args: Optional[dict] = None,
    result_summary: str = "",
) -> None:
    """Record a tool call within an interaction."""
    if _db is None:
        return
    args_json = json.dumps(args or {}, default=str)
    await _db.execute(
        "INSERT INTO interaction_tool_calls (interaction_id, tool_name, args, result_summary, timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        (interaction_id, tool_name, args_json, result_summary, _now_iso()),
    )
    await _db.commit()


async def get_interaction(interaction_id: int) -> Optional[dict]:
    """Fetch a single interaction by id."""
    if _db is None:
        return None
    cursor = await _db.execute("SELECT * FROM interactions WHERE id = ?", (interaction_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None


async def get_interaction_by_ticket(ticket_id: str) -> Optional[dict]:
    """Fetch an interaction by its ServiceNow ticket_id."""
    if _db is None:
        return None
    cursor = await _db.execute(
        "SELECT * FROM interactions WHERE ticket_id = ? ORDER BY id DESC LIMIT 1",
        (ticket_id,),
    )
    row = await cursor.fetchone()
    return dict(row) if row else None
