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

CREATE TABLE IF NOT EXISTS public_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    snippet TEXT NOT NULL DEFAULT '',
    source_domain TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    positive_votes INTEGER NOT NULL DEFAULT 0,
    negative_votes INTEGER NOT NULL DEFAULT 0,
    confidence_score INTEGER NOT NULL DEFAULT 0,
    qdrant_indexed INTEGER NOT NULL DEFAULT 0,
    approval_message_ts TEXT,
    approval_channel TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pub_articles_url ON public_articles(url);
CREATE INDEX IF NOT EXISTS idx_pub_articles_status ON public_articles(status);

CREATE TABLE IF NOT EXISTS article_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL,
    user_slack_id TEXT NOT NULL,
    vote TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (article_id) REFERENCES public_articles(id),
    UNIQUE(article_id, user_slack_id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_form TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    approval_count INTEGER NOT NULL DEFAULT 0,
    denial_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_recommendation_approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER,
    channel_id TEXT NOT NULL,
    thread_ts TEXT NOT NULL DEFAULT '',
    message_ts TEXT NOT NULL,
    original_text TEXT NOT NULL,
    redacted_text TEXT NOT NULL,
    recommendation_ids TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending',
    approval_message_ts TEXT,
    approval_channel TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS collaborative_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER,
    ticket_id TEXT,
    channel_id TEXT NOT NULL,
    thread_ts TEXT NOT NULL DEFAULT '',
    user_slack_id TEXT NOT NULL,
    user_name TEXT NOT NULL DEFAULT '',
    risk_level TEXT NOT NULL DEFAULT 'high',
    trigger_reason TEXT NOT NULL DEFAULT '',
    original_response TEXT NOT NULL,
    issue_summary TEXT NOT NULL DEFAULT '',
    helpdesk_message_ts TEXT,
    placeholder_message_ts TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    resolved_by TEXT,
    modified_response TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_profiles (
    slack_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    device_type TEXT NOT NULL DEFAULT '',
    os TEXT NOT NULL DEFAULT '',
    technical_level TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT '',
    department TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
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

    # Idempotent migration: add refined_text column for recommendation refinement
    try:
        await _db.execute(
            "ALTER TABLE pending_recommendation_approvals ADD COLUMN refined_text TEXT"
        )
        await _db.commit()
    except Exception:
        pass  # Column already exists

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


# ---------------------------------------------------------------------------
# Public articles
# ---------------------------------------------------------------------------

async def create_public_article(
    *,
    url: str,
    title: str,
    content: str = "",
    snippet: str = "",
    source_domain: str,
    status: str = "pending",
) -> int:
    """Insert a public article record. Returns the article id."""
    if _db is None:
        raise RuntimeError("Database not initialized")
    now = _now_iso()
    cursor = await _db.execute(
        """INSERT INTO public_articles
           (url, title, content, snippet, source_domain, status, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (url, title, content, snippet, source_domain, status, now, now),
    )
    await _db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_public_article(article_id: int) -> Optional[dict]:
    """Fetch a single public article by id."""
    if _db is None:
        return None
    cursor = await _db.execute("SELECT * FROM public_articles WHERE id = ?", (article_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None


async def get_public_article_by_url(url: str) -> Optional[dict]:
    """Fetch a public article by its URL."""
    if _db is None:
        return None
    cursor = await _db.execute("SELECT * FROM public_articles WHERE url = ?", (url,))
    row = await cursor.fetchone()
    return dict(row) if row else None


async def update_public_article(article_id: int, **fields: Any) -> None:
    """Update arbitrary fields on a public article record."""
    if _db is None or not fields:
        return
    fields["updated_at"] = _now_iso()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [article_id]
    await _db.execute(
        f"UPDATE public_articles SET {set_clause} WHERE id = ?",  # noqa: S608
        values,
    )
    await _db.commit()


async def record_feedback(
    article_id: int, user_slack_id: str, vote: str,
) -> Optional[dict]:
    """Upsert a feedback vote for an article, recalculate scores, return updated article.

    *vote* should be ``"helpful"`` or ``"not_helpful"``.
    """
    if _db is None:
        return None
    now = _now_iso()
    # Upsert the vote (one vote per user per article)
    await _db.execute(
        """INSERT INTO article_feedback (article_id, user_slack_id, vote, created_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(article_id, user_slack_id) DO UPDATE SET vote = ?, created_at = ?""",
        (article_id, user_slack_id, vote, now, vote, now),
    )
    # Recalculate scores from all votes
    cursor = await _db.execute(
        "SELECT vote, COUNT(*) as cnt FROM article_feedback WHERE article_id = ? GROUP BY vote",
        (article_id,),
    )
    rows = await cursor.fetchall()
    positive = 0
    negative = 0
    for row in rows:
        if row["vote"] == "helpful":
            positive = row["cnt"]
        else:
            negative = row["cnt"]
    confidence = positive - negative
    await _db.execute(
        """UPDATE public_articles
           SET positive_votes = ?, negative_votes = ?, confidence_score = ?, updated_at = ?
           WHERE id = ?""",
        (positive, negative, confidence, now, article_id),
    )
    await _db.commit()
    return await get_public_article(article_id)


async def get_pending_approvals(older_than_minutes: int = 30) -> list[dict]:
    """Return pending articles whose approval request is older than *older_than_minutes*."""
    if _db is None:
        return []
    cursor = await _db.execute(
        """SELECT * FROM public_articles
           WHERE status = 'pending'
             AND approval_message_ts IS NOT NULL
             AND created_at <= datetime('now', ?)""",
        (f"-{older_than_minutes} minutes",),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

async def get_recommendation_by_canonical(canonical_form: str) -> Optional[dict]:
    """Exact lookup of a recommendation by its canonical form."""
    if _db is None:
        return None
    cursor = await _db.execute(
        "SELECT * FROM recommendations WHERE canonical_form = ?", (canonical_form,),
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def create_recommendation(
    canonical_form: str, category: str = "", status: str = "pending",
) -> int:
    """Insert a new recommendation. Returns the recommendation id."""
    if _db is None:
        raise RuntimeError("Database not initialized")
    now = _now_iso()
    cursor = await _db.execute(
        """INSERT INTO recommendations (canonical_form, category, status, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?)""",
        (canonical_form, category, status, now, now),
    )
    await _db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_recommendation(rec_id: int) -> Optional[dict]:
    """Fetch a single recommendation by id."""
    if _db is None:
        return None
    cursor = await _db.execute("SELECT * FROM recommendations WHERE id = ?", (rec_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None


async def update_recommendation(rec_id: int, **fields: Any) -> None:
    """Update arbitrary fields on a recommendation record."""
    if _db is None or not fields:
        return
    fields["updated_at"] = _now_iso()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [rec_id]
    await _db.execute(
        f"UPDATE recommendations SET {set_clause} WHERE id = ?",  # noqa: S608
        values,
    )
    await _db.commit()


async def increment_recommendation_approval(rec_id: int) -> Optional[dict]:
    """Increment approval_count by 1 and return the updated recommendation."""
    if _db is None:
        return None
    now = _now_iso()
    await _db.execute(
        "UPDATE recommendations SET approval_count = approval_count + 1, updated_at = ? WHERE id = ?",
        (now, rec_id),
    )
    await _db.commit()
    return await get_recommendation(rec_id)


async def increment_recommendation_denial(rec_id: int) -> None:
    """Increment denial_count by 1."""
    if _db is None:
        return
    now = _now_iso()
    await _db.execute(
        "UPDATE recommendations SET denial_count = denial_count + 1, updated_at = ? WHERE id = ?",
        (now, rec_id),
    )
    await _db.commit()


async def get_all_recommendations() -> list[dict]:
    """Fetch all recommendations (for semantic matching)."""
    if _db is None:
        return []
    cursor = await _db.execute("SELECT * FROM recommendations")
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Pending recommendation approvals
# ---------------------------------------------------------------------------

async def create_pending_rec_approval(
    *,
    channel_id: str,
    message_ts: str,
    original_text: str,
    redacted_text: str,
    recommendation_ids: list[int],
    thread_ts: str = "",
    interaction_id: Optional[int] = None,
) -> int:
    """Insert a pending recommendation approval. Returns its id."""
    if _db is None:
        raise RuntimeError("Database not initialized")
    now = _now_iso()
    ids_json = json.dumps(recommendation_ids)
    cursor = await _db.execute(
        """INSERT INTO pending_recommendation_approvals
           (interaction_id, channel_id, thread_ts, message_ts, original_text,
            redacted_text, recommendation_ids, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (interaction_id, channel_id, thread_ts, message_ts, original_text,
         redacted_text, ids_json, now, now),
    )
    await _db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_pending_rec_approval(approval_id: int) -> Optional[dict]:
    """Fetch a single pending recommendation approval by id."""
    if _db is None:
        return None
    cursor = await _db.execute(
        "SELECT * FROM pending_recommendation_approvals WHERE id = ?", (approval_id,),
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def update_pending_rec_approval(approval_id: int, **fields: Any) -> None:
    """Update arbitrary fields on a pending recommendation approval."""
    if _db is None or not fields:
        return
    fields["updated_at"] = _now_iso()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [approval_id]
    await _db.execute(
        f"UPDATE pending_recommendation_approvals SET {set_clause} WHERE id = ?",  # noqa: S608
        values,
    )
    await _db.commit()


async def get_expired_rec_approvals(older_than_minutes: int = 30) -> list[dict]:
    """Return pending recommendation approvals older than the given threshold."""
    if _db is None:
        return []
    cursor = await _db.execute(
        """SELECT * FROM pending_recommendation_approvals
           WHERE status = 'pending'
             AND created_at <= datetime('now', ?)""",
        (f"-{older_than_minutes} minutes",),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Collaborative reviews
# ---------------------------------------------------------------------------

async def create_collaborative_review(
    *,
    channel_id: str,
    thread_ts: str = "",
    user_slack_id: str,
    user_name: str = "",
    original_response: str,
    trigger_reason: str = "",
    issue_summary: str = "",
    risk_level: str = "high",
    interaction_id: Optional[int] = None,
    ticket_id: Optional[str] = None,
    placeholder_message_ts: Optional[str] = None,
) -> int:
    """Insert a new collaborative review record. Returns the review id."""
    if _db is None:
        raise RuntimeError("Database not initialized")
    now = _now_iso()
    cursor = await _db.execute(
        """INSERT INTO collaborative_reviews
           (interaction_id, ticket_id, channel_id, thread_ts,
            user_slack_id, user_name, risk_level, trigger_reason,
            original_response, issue_summary, placeholder_message_ts,
            created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            interaction_id, ticket_id, channel_id, thread_ts,
            user_slack_id, user_name, risk_level, trigger_reason,
            original_response, issue_summary, placeholder_message_ts,
            now, now,
        ),
    )
    await _db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_collaborative_review(review_id: int) -> Optional[dict]:
    """Fetch a single collaborative review by id."""
    if _db is None:
        return None
    cursor = await _db.execute(
        "SELECT * FROM collaborative_reviews WHERE id = ?", (review_id,),
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def update_collaborative_review(review_id: int, **fields: Any) -> None:
    """Update arbitrary fields on a collaborative review record."""
    if _db is None or not fields:
        return
    fields["updated_at"] = _now_iso()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [review_id]
    await _db.execute(
        f"UPDATE collaborative_reviews SET {set_clause} WHERE id = ?",  # noqa: S608
        values,
    )
    await _db.commit()


async def get_collaborative_review_by_ticket(ticket_id: str) -> Optional[dict]:
    """Fetch the most recent collaborative review for a ticket."""
    if _db is None:
        return None
    cursor = await _db.execute(
        "SELECT * FROM collaborative_reviews WHERE ticket_id = ? ORDER BY id DESC LIMIT 1",
        (ticket_id,),
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def get_expired_collaborative_reviews(older_than_minutes: int = 30) -> list[dict]:
    """Return pending collaborative reviews older than the given threshold."""
    if _db is None:
        return []
    cursor = await _db.execute(
        """SELECT * FROM collaborative_reviews
           WHERE status = 'pending'
             AND created_at <= datetime('now', ?)""",
        (f"-{older_than_minutes} minutes",),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# User profiles
# ---------------------------------------------------------------------------

async def get_user_profile(slack_id: str) -> Optional[dict]:
    """Fetch a user profile by Slack ID."""
    if _db is None:
        return None
    cursor = await _db.execute(
        "SELECT * FROM user_profiles WHERE slack_id = ?", (slack_id,),
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def upsert_user_profile(slack_id: str, **fields: Any) -> None:
    """Create or update a user profile, merging non-empty fields.

    For the ``notes`` field, new text is appended rather than overwriting.
    Empty-string values are ignored so they don't clobber existing data.
    """
    if _db is None:
        return

    existing = await get_user_profile(slack_id)
    now = _now_iso()

    if existing:
        updates: dict[str, Any] = {}
        for key in ("display_name", "device_type", "os", "technical_level",
                     "role", "department"):
            new_val = fields.get(key, "")
            if new_val:
                updates[key] = new_val
        # Append notes rather than overwrite
        new_notes = fields.get("notes", "")
        if new_notes:
            old_notes = existing.get("notes", "")
            if old_notes:
                updates["notes"] = f"{old_notes}; {new_notes}"
            else:
                updates["notes"] = new_notes
        if not updates:
            return
        updates["updated_at"] = now
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [slack_id]
        await _db.execute(
            f"UPDATE user_profiles SET {set_clause} WHERE slack_id = ?",  # noqa: S608
            values,
        )
    else:
        profile = {
            "slack_id": slack_id,
            "display_name": fields.get("display_name", ""),
            "device_type": fields.get("device_type", ""),
            "os": fields.get("os", ""),
            "technical_level": fields.get("technical_level", ""),
            "role": fields.get("role", ""),
            "department": fields.get("department", ""),
            "notes": fields.get("notes", ""),
            "updated_at": now,
        }
        cols = ", ".join(profile.keys())
        placeholders = ", ".join("?" for _ in profile)
        await _db.execute(
            f"INSERT INTO user_profiles ({cols}) VALUES ({placeholders})",  # noqa: S608
            list(profile.values()),
        )

    await _db.commit()
