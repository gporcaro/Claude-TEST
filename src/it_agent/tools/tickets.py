"""Ticket tool wrappers — bridge between agent executor and DB queries."""

from __future__ import annotations

from it_agent.config import Settings
from it_agent.db import queries
from it_agent.db.models import Ticket, TicketComment


async def create_ticket(
    title: str,
    description: str,
    priority: str = "medium",
    category: str = "",
    _settings: Settings | None = None,
    _user_id: str = "unknown",
    **_,
) -> dict:
    """Create a new IT support ticket."""
    if _settings is None:
        return {"error": "Settings not configured"}

    ticket = Ticket(
        title=title,
        description=description,
        priority=priority,
        category=category,
        requester_id=_user_id,
    )
    ticket = await queries.create_ticket(_settings.db_path, ticket)
    return {"success": True, "ticket": ticket.to_dict()}


async def get_ticket(
    ticket_id: int,
    _settings: Settings | None = None,
    **_,
) -> dict:
    """Retrieve a ticket by ID."""
    if _settings is None:
        return {"error": "Settings not configured"}

    ticket = await queries.get_ticket(_settings.db_path, ticket_id)
    if ticket is None:
        return {"error": f"Ticket #{ticket_id} not found"}

    comments = await queries.get_comments(_settings.db_path, ticket_id)
    return {
        "ticket": ticket.to_dict(),
        "comments": [c.to_dict() for c in comments],
    }


async def update_ticket(
    ticket_id: int,
    status: str | None = None,
    priority: str | None = None,
    assignee_id: str | None = None,
    comment: str | None = None,
    _settings: Settings | None = None,
    _user_id: str = "unknown",
    **_,
) -> dict:
    """Update ticket fields and/or add a comment."""
    if _settings is None:
        return {"error": "Settings not configured"}

    # Update ticket fields
    ticket = await queries.update_ticket(
        _settings.db_path,
        ticket_id,
        status=status,
        priority=priority,
        assignee_id=assignee_id,
    )
    if ticket is None:
        return {"error": f"Ticket #{ticket_id} not found"}

    # Add comment if provided
    if comment:
        tc = TicketComment(
            ticket_id=ticket_id,
            author_id=_user_id,
            content=comment,
        )
        await queries.add_comment(_settings.db_path, tc)

    return {"success": True, "ticket": ticket.to_dict()}


async def list_tickets(
    status: str | None = None,
    priority: str | None = None,
    requester_id: str | None = None,
    limit: int = 10,
    _settings: Settings | None = None,
    **_,
) -> dict:
    """List tickets with optional filters."""
    if _settings is None:
        return {"error": "Settings not configured"}

    tickets = await queries.list_tickets(
        _settings.db_path,
        status=status,
        priority=priority,
        requester_id=requester_id,
        limit=limit,
    )
    return {"tickets": [t.to_dict() for t in tickets], "count": len(tickets)}
