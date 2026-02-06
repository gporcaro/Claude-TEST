"""aiosqlite connection management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    priority TEXT NOT NULL DEFAULT 'medium',
    category TEXT NOT NULL DEFAULT '',
    requester_id TEXT NOT NULL DEFAULT '',
    assignee_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS ticket_comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id INTEGER NOT NULL,
    author_id TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (ticket_id) REFERENCES tickets(id)
);
"""


async def init_db(db_path: Path) -> None:
    """Initialize the database and create tables."""
    async with aiosqlite.connect(str(db_path)) as db:
        await db.executescript(_CREATE_TABLES_SQL)
        await db.commit()
    logger.info("Database initialized at %s", db_path)


@asynccontextmanager
async def get_db(db_path: Path):
    """Async context manager for database connections."""
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()
