# IT Support AI Agent

An AI-powered IT support agent that lives in Slack, using Claude as its backbone. It provides knowledge base Q&A, ticket management, and system diagnostics through natural language conversation.

## Features

- **Diagnostics** — Ping hosts, DNS lookups, disk usage checks, service status monitoring
- **Ticket Management** — Create, view, update, and list IT support tickets (SQLite-backed)
- **Knowledge Base** — Semantic search over internal IT documentation (ChromaDB)
- **Conversational** — Per-thread conversation history for contextual follow-ups

## Architecture

```
User in Slack → Socket Mode → handlers.py → Agent core.py (tool loop) → Claude API
                                                    ↓
                                      Tool Executor dispatches to:
                                      ├── diagnostics.py (ping, DNS, disk, services)
                                      ├── tickets.py     (CRUD via SQLite)
                                      └── knowledge.py   (semantic search via ChromaDB)
                                                    ↓
                                      Results → Claude → Slack response
```

## Setup

### 1. Prerequisites
- Python 3.11+
- A Slack workspace with a bot app configured for Socket Mode
- An Anthropic API key

### 2. Slack App Configuration
1. Create a new Slack app at https://api.slack.com/apps
2. Enable **Socket Mode** (generates an app-level token starting with `xapp-`)
3. Add **Bot Token Scopes**: `app_mentions:read`, `chat:write`, `im:history`, `im:read`, `im:write`
4. Subscribe to **Events**: `app_mention`, `message.im`
5. Install the app to your workspace (generates a bot token starting with `xoxb-`)

### 3. Install

```bash
cd Claude-TEST
cp .env.example .env
# Edit .env with your tokens

pip install -e ".[dev]"
```

### 4. Index Knowledge Base

```bash
it-agent-index --docs src/it_agent/docs --chroma chroma_data
```

### 5. Run

```bash
it-agent
```

## Usage

Mention the bot in any channel or send it a DM:

- `@IT Agent How do I set up VPN?` — searches knowledge base
- `@IT Agent Ping google.com` — runs diagnostics
- `@IT Agent Create a ticket for my broken monitor` — creates a ticket
- `@IT Agent Show me ticket #3` — retrieves ticket details
- `@IT Agent List open tickets` — lists filtered tickets

## Development

```bash
# Lint
ruff check src/

# Format
ruff format src/

# Test
pytest
```

## Project Structure

```
src/it_agent/
├── config.py          # Pydantic Settings (.env loading)
├── main.py            # Entry point
├── bot/               # Slack integration
├── agent/             # Claude tool loop
├── tools/             # Tool implementations
├── knowledge/         # ChromaDB indexer + search
├── db/                # SQLite ticket storage
└── docs/              # Knowledge base markdown files
```
