"""Ticket and Comment dataclasses + enums."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    WAITING = "waiting"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Ticket:
    id: int | None = None
    title: str = ""
    description: str = ""
    status: str = TicketStatus.OPEN
    priority: str = TicketPriority.MEDIUM
    category: str = ""
    requester_id: str = ""
    assignee_id: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    resolved_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TicketComment:
    id: int | None = None
    ticket_id: int = 0
    author_id: str = ""
    content: str = ""
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)
