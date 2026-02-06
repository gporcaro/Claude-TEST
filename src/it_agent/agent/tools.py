"""JSON schema definitions for the 9 agent tools."""

TOOLS = [
    # --- Diagnostics ---
    {
        "name": "ping_host",
        "description": (
            "Ping a hostname or IP address to check if it is reachable. Returns latency info."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": (
                        "Hostname or IP address to ping (e.g. 'google.com' or '8.8.8.8')"
                    ),
                },
                "count": {
                    "type": "integer",
                    "description": "Number of ping packets to send (default 4, max 10)",
                    "default": 4,
                },
            },
            "required": ["host"],
        },
    },
    {
        "name": "dns_lookup",
        "description": (
            "Perform a DNS lookup on a hostname. Returns resolved IP addresses and record info."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hostname": {
                    "type": "string",
                    "description": "Hostname to resolve (e.g. 'google.com')",
                },
                "record_type": {
                    "type": "string",
                    "description": "DNS record type (A, AAAA, MX, CNAME, TXT, NS)",
                    "default": "A",
                    "enum": ["A", "AAAA", "MX", "CNAME", "TXT", "NS"],
                },
            },
            "required": ["hostname"],
        },
    },
    {
        "name": "check_disk_usage",
        "description": "Check disk usage on the system. Returns filesystem usage info.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Filesystem path to check (default '/')",
                    "default": "/",
                },
            },
            "required": [],
        },
    },
    {
        "name": "check_service_status",
        "description": "Check if a system service or process is running.",
        "input_schema": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": (
                        "Name of the service or process to check (e.g. 'nginx', 'postgres')"
                    ),
                },
            },
            "required": ["service_name"],
        },
    },
    # --- Tickets ---
    {
        "name": "create_ticket",
        "description": "Create a new IT support ticket.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short title for the ticket"},
                "description": {
                    "type": "string",
                    "description": "Detailed description of the issue",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "default": "medium",
                },
                "category": {
                    "type": "string",
                    "description": "Category (e.g. 'network', 'hardware', 'software', 'access')",
                },
            },
            "required": ["title", "description"],
        },
    },
    {
        "name": "get_ticket",
        "description": "Retrieve an IT support ticket by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "integer", "description": "The ticket ID"},
            },
            "required": ["ticket_id"],
        },
    },
    {
        "name": "update_ticket",
        "description": (
            "Update an existing IT support ticket."
            " Can change status, priority, assignee, or add a comment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "integer", "description": "The ticket ID to update"},
                "status": {
                    "type": "string",
                    "enum": ["open", "in_progress", "waiting", "resolved", "closed"],
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                },
                "assignee_id": {"type": "string", "description": "Slack user ID of assignee"},
                "comment": {"type": "string", "description": "Comment to add to the ticket"},
            },
            "required": ["ticket_id"],
        },
    },
    {
        "name": "list_tickets",
        "description": "List IT support tickets with optional filters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["open", "in_progress", "waiting", "resolved", "closed"],
                    "description": "Filter by status",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "description": "Filter by priority",
                },
                "requester_id": {
                    "type": "string",
                    "description": "Filter by requester Slack user ID",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of tickets to return (default 10)",
                    "default": 10,
                },
            },
            "required": [],
        },
    },
    # --- Knowledge Base ---
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the IT knowledge base using semantic search."
            " Use this to find solutions, documentation, and procedures"
            " for common IT issues."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query (e.g. 'how to set up VPN')",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 3, max 10)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
]
