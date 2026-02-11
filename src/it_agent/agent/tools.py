"""JSON schema definitions for the 9 agent tools."""

TOOLS = [
    # --- Diagnostics ---
    {
        "name": "ping_host",
        "description": (
            "Ping a hostname or IP address to check if it is reachable. Returns latency info."
        ),
        "parameters": {
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
        "parameters": {
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
        "parameters": {
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
        "parameters": {
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
        "parameters": {
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
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "ServiceNow incident number (e.g. 'INC0010001')",
                },
            },
            "required": ["ticket_id"],
        },
    },
    {
        "name": "update_ticket",
        "description": (
            "Update an existing IT support ticket."
            " Can change status, priority, assignee, requester, or add a comment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "ServiceNow incident number (e.g. 'INC0010001')",
                },
                "status": {
                    "type": "string",
                    "enum": ["open", "in_progress", "waiting", "resolved", "closed"],
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                },
                "assignee_id": {
                    "type": "string",
                    "description": (
                        "ServiceNow sys_id of the assignee. Use the lookup_user tool "
                        "first to resolve a person's name to their servicenow_sys_id."
                    ),
                },
                "requester_id": {
                    "type": "string",
                    "description": (
                        "ServiceNow sys_id of the requester (caller). Use the lookup_user "
                        "tool first to resolve a person's name to their servicenow_sys_id."
                    ),
                },
                "comment": {"type": "string", "description": "Comment to add to the ticket"},
                "close_notes": {
                    "type": "string",
                    "description": (
                        "Resolution notes (required when setting status to"
                        " 'resolved' or 'closed')"
                    ),
                },
            },
            "required": ["ticket_id"],
        },
    },
    {
        "name": "list_tickets",
        "description": "List IT support tickets with optional filters.",
        "parameters": {
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
        "parameters": {
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
    # --- Public Articles ---
    {
        "name": "search_public_articles",
        "description": (
            "Search for articles from official vendor sources (Apple, Microsoft, Dell) "
            "about hardware, software, and OS topics. Use this when the internal knowledge "
            "base doesn't have relevant results for device-specific issues, OS configuration, "
            "or hardware troubleshooting."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results (default 3, max 5)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    # --- User Lookup ---
    {
        "name": "lookup_user",
        "description": (
            "Search for a user by name (partial or full) across Slack and ServiceNow. "
            "Returns matching candidates with their Slack user ID, ServiceNow sys_id, "
            "display name, and email. Use this when you need to assign a ticket to "
            "someone by name, or identify a user mentioned in conversation. "
            "If multiple matches are found, ask the requester to confirm which user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Name to search for (e.g. 'John', 'John Smith', 'jsmith')"
                    ),
                },
            },
            "required": ["name"],
        },
    },
]
