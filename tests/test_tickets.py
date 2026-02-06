"""Tests for ticket system."""

from __future__ import annotations

import pytest

from it_agent.db import queries
from it_agent.db.database import init_db
from it_agent.db.models import Ticket, TicketComment, TicketPriority, TicketStatus


@pytest.fixture
async def db_path(tmp_path):
    """Create a temporary database for testing."""
    path = tmp_path / "test_tickets.db"
    await init_db(path)
    return path


@pytest.mark.asyncio
async def test_create_and_get_ticket(db_path):
    ticket = Ticket(
        title="Test Issue",
        description="Something is broken",
        priority=TicketPriority.HIGH,
        category="software",
        requester_id="U123",
    )
    created = await queries.create_ticket(db_path, ticket)
    assert created.id is not None
    assert created.id == 1

    fetched = await queries.get_ticket(db_path, created.id)
    assert fetched is not None
    assert fetched.title == "Test Issue"
    assert fetched.priority == "high"
    assert fetched.status == "open"


@pytest.mark.asyncio
async def test_get_nonexistent_ticket(db_path):
    result = await queries.get_ticket(db_path, 9999)
    assert result is None


@pytest.mark.asyncio
async def test_update_ticket_status(db_path):
    ticket = Ticket(title="Update Test", description="desc", requester_id="U123")
    created = await queries.create_ticket(db_path, ticket)

    updated = await queries.update_ticket(db_path, created.id, status=TicketStatus.IN_PROGRESS)
    assert updated.status == "in_progress"


@pytest.mark.asyncio
async def test_update_ticket_resolved_sets_resolved_at(db_path):
    ticket = Ticket(title="Resolve Test", description="desc")
    created = await queries.create_ticket(db_path, ticket)

    updated = await queries.update_ticket(db_path, created.id, status=TicketStatus.RESOLVED)
    assert updated.resolved_at is not None


@pytest.mark.asyncio
async def test_list_tickets(db_path):
    for i in range(5):
        await queries.create_ticket(
            db_path,
            Ticket(title=f"Ticket {i}", description="desc", priority="medium"),
        )

    tickets = await queries.list_tickets(db_path)
    assert len(tickets) == 5


@pytest.mark.asyncio
async def test_list_tickets_filter_by_status(db_path):
    await queries.create_ticket(db_path, Ticket(title="Open", description="d", status="open"))
    await queries.create_ticket(db_path, Ticket(title="Closed", description="d", status="closed"))

    open_tickets = await queries.list_tickets(db_path, status="open")
    assert len(open_tickets) == 1
    assert open_tickets[0].title == "Open"


@pytest.mark.asyncio
async def test_list_tickets_limit(db_path):
    for i in range(10):
        await queries.create_ticket(db_path, Ticket(title=f"T{i}", description="d"))

    tickets = await queries.list_tickets(db_path, limit=3)
    assert len(tickets) == 3


@pytest.mark.asyncio
async def test_add_and_get_comments(db_path):
    ticket = Ticket(title="Comment Test", description="desc")
    created = await queries.create_ticket(db_path, ticket)

    comment = TicketComment(ticket_id=created.id, author_id="U456", content="Working on it")
    saved = await queries.add_comment(db_path, comment)
    assert saved.id is not None

    comments = await queries.get_comments(db_path, created.id)
    assert len(comments) == 1
    assert comments[0].content == "Working on it"
