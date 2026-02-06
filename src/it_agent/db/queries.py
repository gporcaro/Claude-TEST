"""Ticket CRUD operations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from it_agent.db.database import get_db
from it_agent.db.models import Ticket, TicketComment, TicketStatus


async def create_ticket(db_path: Path, ticket: Ticket) -> Ticket:
    """Create a new ticket and return it with its ID."""
    async with get_db(db_path) as db:
        cursor = await db.execute(
            """INSERT INTO tickets (title, description, status, priority, category,
                                    requester_id, assignee_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ticket.title,
                ticket.description,
                ticket.status,
                ticket.priority,
                ticket.category,
                ticket.requester_id,
                ticket.assignee_id,
                ticket.created_at,
                ticket.updated_at,
            ),
        )
        await db.commit()
        ticket.id = cursor.lastrowid
    return ticket


async def get_ticket(db_path: Path, ticket_id: int) -> Ticket | None:
    """Get a ticket by ID."""
    async with get_db(db_path) as db:
        cursor = await db.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return Ticket(**dict(row))


async def update_ticket(db_path: Path, ticket_id: int, **updates) -> Ticket | None:
    """Update a ticket's fields."""
    allowed = {"status", "priority", "category", "assignee_id"}
    fields = {k: v for k, v in updates.items() if k in allowed and v is not None}

    if not fields:
        return await get_ticket(db_path, ticket_id)

    now = datetime.now(timezone.utc).isoformat()
    fields["updated_at"] = now

    # Set resolved_at if status is resolved
    if fields.get("status") == TicketStatus.RESOLVED:
        fields["resolved_at"] = now

    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [ticket_id]

    async with get_db(db_path) as db:
        await db.execute(
            f"UPDATE tickets SET {set_clause} WHERE id = ?",  # noqa: S608
            values,
        )
        await db.commit()

    return await get_ticket(db_path, ticket_id)


async def list_tickets(
    db_path: Path,
    status: str | None = None,
    priority: str | None = None,
    requester_id: str | None = None,
    limit: int = 10,
) -> list[Ticket]:
    """List tickets with optional filters."""
    query = "SELECT * FROM tickets WHERE 1=1"
    params: list = []

    if status:
        query += " AND status = ?"
        params.append(status)
    if priority:
        query += " AND priority = ?"
        params.append(priority)
    if requester_id:
        query += " AND requester_id = ?"
        params.append(requester_id)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(min(limit, 50))

    async with get_db(db_path) as db:
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [Ticket(**dict(row)) for row in rows]


async def add_comment(db_path: Path, comment: TicketComment) -> TicketComment:
    """Add a comment to a ticket."""
    async with get_db(db_path) as db:
        cursor = await db.execute(
            """INSERT INTO ticket_comments (ticket_id, author_id, content, created_at)
               VALUES (?, ?, ?, ?)""",
            (comment.ticket_id, comment.author_id, comment.content, comment.created_at),
        )
        await db.commit()
        comment.id = cursor.lastrowid
    return comment


async def get_comments(db_path: Path, ticket_id: int) -> list[TicketComment]:
    """Get all comments for a ticket."""
    async with get_db(db_path) as db:
        cursor = await db.execute(
            "SELECT * FROM ticket_comments WHERE ticket_id = ? ORDER BY created_at",
            (ticket_id,),
        )
        rows = await cursor.fetchall()
        return [TicketComment(**dict(row)) for row in rows]
