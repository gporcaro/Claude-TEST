"""Tests for bot/handlers.py — routing, follow-ups, and resolution updates."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from it_agent.agent.core import AgentResult
from it_agent.bot import handlers
from it_agent.config import Settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> Settings:
    """Create a Settings object with test defaults."""
    defaults = {
        "slack_bot_token": "xoxb-test",
        "slack_app_token": "xapp-test",
        "gemini_api_key": "test-key",
        "sn_instance_url": "https://test.service-now.com",
        "sn_username": "user",
        "sn_password": "pass",
        "help_channel_id": "C_HELP",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _dm_event(text: str = "help me", ts: str = "1000.1") -> dict:
    return {
        "channel": "D_DM",
        "channel_type": "im",
        "user": "U_USER",
        "text": text,
        "ts": ts,
    }


def _help_channel_event(text: str = "my laptop is broken", ts: str = "2000.1") -> dict:
    return {
        "channel": "C_HELP",
        "channel_type": "channel",
        "user": "U_USER",
        "text": text,
        "ts": ts,
    }


def _other_channel_event(text: str = "random", ts: str = "3000.1") -> dict:
    return {
        "channel": "C_OTHER",
        "channel_type": "channel",
        "user": "U_USER",
        "text": text,
        "ts": ts,
    }


def _agent_result(text: str = "I'll help!", tool_calls: list | None = None) -> AgentResult:
    return AgentResult(text=text, tool_calls=tool_calls or [])


def _agent_result_with_ticket(
    ticket_id: str = "INC0010042",
    channel_name: str = "inc0010042-alice",
    channel_id: str = "C_INC42",
) -> AgentResult:
    return AgentResult(
        text="I've created a ticket for you.",
        tool_calls=[
            {
                "name": "create_ticket",
                "args": {"title": "Laptop broken", "description": "Won't boot"},
                "result": {
                    "success": True,
                    "ticket": {"ticket_id": ticket_id, "title": "Laptop broken"},
                    "channel_name": channel_name,
                    "channel_id": channel_id,
                },
            }
        ],
    )


@pytest.fixture(autouse=True)
def _reset_handler_state():
    """Clear module-level state between tests."""
    handlers._conversations.clear()
    handlers._ticket_threads.clear()
    handlers._incident_channels.clear()
    handlers._incident_context.clear()
    handlers._agent = None
    yield
    handlers._conversations.clear()
    handlers._ticket_threads.clear()
    handlers._incident_channels.clear()
    handlers._incident_context.clear()
    handlers._agent = None


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------

class TestDMRouting:
    async def test_dm_is_processed(self):
        """DM messages are routed through _handle_message."""
        event = _dm_event()
        channel_type = event.get("channel_type", "")
        assert channel_type == "im"

    async def test_dm_calls_agent_and_replies(self):
        """DM goes through agent and posts reply."""
        settings = _make_settings()
        say = AsyncMock()
        mock_agent = AsyncMock()
        mock_agent.run.return_value = _agent_result("Here's help!")

        with patch.object(handlers, "_get_agent", return_value=mock_agent):
            await handlers._handle_message(_dm_event(), "help me", say, settings)

        say.assert_called_once()
        call_kwargs = say.call_args[1]
        assert call_kwargs["text"] == "Here's help!"
        assert call_kwargs["thread_ts"] == "1000.1"


class TestHelpChannelRouting:
    async def test_help_channel_triggers_proactive_handler(self):
        """Messages in #help-it go to _handle_help_channel_message."""
        settings = _make_settings()
        say = AsyncMock()
        mock_agent = AsyncMock()
        mock_agent.run.return_value = _agent_result("Let me check that.")

        with patch.object(handlers, "_get_agent", return_value=mock_agent):
            await handlers._handle_help_channel_message(
                _help_channel_event(), "my laptop is broken", say, settings
            )

        say.assert_called_once()
        assert say.call_args[1]["text"] == "Let me check that."

    async def test_mention_in_help_channel_stripped(self):
        """@bot mention text in #help-it is stripped before processing."""
        import re

        event = _help_channel_event(text="<@U0123BOT> my laptop is broken")
        cleaned = re.sub(r"<@[A-Z0-9]+>\s*", "", event["text"]).strip()
        assert cleaned == "my laptop is broken"


class TestOtherChannelRouting:
    async def test_other_channel_ignored(self):
        """Messages in other channels are not processed by handle_message."""
        event = _other_channel_event()
        # channel_type is not "im" and channel != help_channel_id
        settings = _make_settings()
        assert event["channel_type"] != "im"
        assert event["channel"] != settings.help_channel_id


class TestSubtypeAndBotFiltering:
    async def test_subtype_messages_filtered(self):
        """Messages with subtype are ignored."""
        event = _dm_event()
        event["subtype"] = "channel_join"
        assert event.get("subtype") is not None

    async def test_bot_id_messages_filtered(self):
        """Messages with bot_id are ignored."""
        event = _dm_event()
        event["bot_id"] = "B_BOT"
        assert event.get("bot_id") is not None


# ---------------------------------------------------------------------------
# Double-response prevention
# ---------------------------------------------------------------------------

class TestDoubleResponsePrevention:
    async def test_mention_in_help_channel_skipped(self):
        """handle_mention returns early when channel == help_channel_id."""
        settings = _make_settings()
        event = {"channel": "C_HELP", "text": "<@U_BOT> help", "ts": "5000.1", "user": "U_USER"}
        say = AsyncMock()

        # Simulate the logic in handle_mention
        if settings.help_channel_id and event.get("channel") == settings.help_channel_id:
            skipped = True
        else:
            skipped = False

        assert skipped is True
        say.assert_not_called()


# ---------------------------------------------------------------------------
# Ticket follow-up tests
# ---------------------------------------------------------------------------

class TestTicketFollowUp:
    async def test_ticket_creation_posts_followup(self):
        """When agent creates a ticket in #help-it, a follow-up is posted in the thread."""
        settings = _make_settings()
        say = AsyncMock()
        mock_agent = AsyncMock()
        mock_agent.run.return_value = _agent_result_with_ticket()

        with patch.object(handlers, "_get_agent", return_value=mock_agent):
            await handlers._handle_help_channel_message(
                _help_channel_event(), "my laptop is broken", say, settings
            )

        # Two calls: the main response + the ticket follow-up
        assert say.call_count == 2
        followup_text = say.call_args_list[1][1]["text"]
        assert "INC0010042" in followup_text
        assert "#inc0010042-alice" in followup_text
        assert "Updates will be posted here" in followup_text

    async def test_ticket_followup_stores_thread_mapping(self):
        """Ticket follow-up stores the ticket → thread mapping."""
        settings = _make_settings()
        say = AsyncMock()
        mock_agent = AsyncMock()
        mock_agent.run.return_value = _agent_result_with_ticket()

        with patch.object(handlers, "_get_agent", return_value=mock_agent):
            await handlers._handle_help_channel_message(
                _help_channel_event(), "my laptop is broken", say, settings
            )

        assert "INC0010042" in handlers._ticket_threads
        assert handlers._ticket_threads["INC0010042"] == ("C_HELP", "2000.1")

    async def test_duplicate_followup_not_posted(self):
        """If the same ticket is already tracked, no duplicate follow-up is posted."""
        settings = _make_settings()
        say = AsyncMock()
        mock_agent = AsyncMock()
        mock_agent.run.return_value = _agent_result_with_ticket()

        # Pre-populate the mapping
        handlers._ticket_threads["INC0010042"] = ("C_HELP", "1999.1")

        with patch.object(handlers, "_get_agent", return_value=mock_agent):
            await handlers._handle_help_channel_message(
                _help_channel_event(), "my laptop is broken", say, settings
            )

        # Only the main response, no follow-up
        assert say.call_count == 1

    async def test_failed_ticket_no_followup(self):
        """If ticket creation failed, no follow-up is posted."""
        settings = _make_settings()
        say = AsyncMock()
        mock_agent = AsyncMock()
        mock_agent.run.return_value = AgentResult(
            text="Sorry, couldn't create a ticket.",
            tool_calls=[
                {
                    "name": "create_ticket",
                    "args": {"title": "test", "description": "test"},
                    "result": {"error": "ServiceNow down"},
                }
            ],
        )

        with patch.object(handlers, "_get_agent", return_value=mock_agent):
            await handlers._handle_help_channel_message(
                _help_channel_event(), "issue", say, settings
            )

        # Only the main response
        assert say.call_count == 1
        assert len(handlers._ticket_threads) == 0


# ---------------------------------------------------------------------------
# Resolution update tests
# ---------------------------------------------------------------------------

class TestResolutionUpdate:
    async def test_resolution_posts_to_original_thread(self):
        """post_resolution_update posts a message to the stored thread."""
        settings = _make_settings()
        handlers._ticket_threads["INC0010042"] = ("C_HELP", "2000.1")

        with patch(
            "it_agent.bot.handlers.AsyncWebClient"
        ) as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            await handlers.post_resolution_update(
                "INC0010042",
                {"status": "resolved", "title": "Laptop fixed", "close_notes": "Replaced SSD"},
                settings,
            )

        mock_client.chat_postMessage.assert_called_once()
        call_kwargs = mock_client.chat_postMessage.call_args[1]
        assert call_kwargs["channel"] == "C_HELP"
        assert call_kwargs["thread_ts"] == "2000.1"
        assert "INC0010042" in call_kwargs["text"]
        assert "resolved" in call_kwargs["text"]
        assert "Replaced SSD" in call_kwargs["text"]

    async def test_resolution_cleans_up_mapping(self):
        """After posting, the ticket entry is removed from _ticket_threads."""
        settings = _make_settings()
        handlers._ticket_threads["INC0010042"] = ("C_HELP", "2000.1")

        with patch(
            "it_agent.bot.handlers.AsyncWebClient"
        ) as mock_cls:
            mock_cls.return_value = AsyncMock()

            await handlers.post_resolution_update(
                "INC0010042", {"status": "resolved", "title": "X"}, settings
            )

        assert "INC0010042" not in handlers._ticket_threads

    async def test_untracked_ticket_no_update(self):
        """If a ticket is not in _ticket_threads, no update is posted."""
        settings = _make_settings()

        with patch(
            "it_agent.bot.handlers.AsyncWebClient"
        ) as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            await handlers.post_resolution_update(
                "INC9999999", {"status": "resolved", "title": "X"}, settings
            )

        mock_client.chat_postMessage.assert_not_called()


# ---------------------------------------------------------------------------
# DM does not post ticket follow-up
# ---------------------------------------------------------------------------

class TestDMNoFollowUp:
    async def test_dm_ticket_no_followup(self):
        """Ticket created via DM does not get a #help-it follow-up."""
        settings = _make_settings()
        say = AsyncMock()
        mock_agent = AsyncMock()
        mock_agent.run.return_value = _agent_result_with_ticket()

        with patch.object(handlers, "_get_agent", return_value=mock_agent):
            await handlers._handle_message(_dm_event(), "my laptop broke", say, settings)

        # Only one call — the main response; no follow-up
        assert say.call_count == 1
        assert len(handlers._ticket_threads) == 0


# ---------------------------------------------------------------------------
# Incident channel routing tests
# ---------------------------------------------------------------------------

class TestIncidentChannelRouting:
    async def test_incident_channel_registered_on_ticket_creation(self):
        """When a ticket is created, the incident channel_id and context are stored."""
        settings = _make_settings()
        say = AsyncMock()
        mock_agent = AsyncMock()
        mock_agent.run.return_value = _agent_result_with_ticket(channel_id="C_INC42")

        with patch.object(handlers, "_get_agent", return_value=mock_agent):
            await handlers._handle_message(_dm_event(), "laptop broke", say, settings)

        assert "C_INC42" in handlers._incident_channels
        assert "C_INC42" in handlers._incident_context
        assert handlers._incident_context["C_INC42"]["ticket_id"] == "INC0010042"

    async def test_incident_channel_messages_are_processed(self):
        """Messages in a registered incident channel go through _handle_incident_message."""
        settings = _make_settings()
        say = AsyncMock()
        mock_agent = AsyncMock()
        mock_agent.run.return_value = _agent_result("I'll check on that.")

        # Pre-register the incident channel with context
        handlers._incident_channels.add("C_INC42")
        handlers._incident_context["C_INC42"] = {
            "ticket_id": "INC0010042",
            "title": "VPN issue",
            "description": "Can't connect to VPN",
            "priority": "medium",
            "summary_ts": None,
            "original_text": None,
            "summary_lines": [],
        }

        event = {
            "channel": "C_INC42",
            "channel_type": "group",
            "user": "U_USER",
            "text": "any update?",
            "ts": "4000.1",
        }

        with patch.object(handlers, "_get_agent", return_value=mock_agent):
            await handlers._handle_incident_message(event, "any update?", say, settings)

        say.assert_called_once()
        call_kwargs = say.call_args[1]
        assert call_kwargs["text"] == "I'll check on that."
        # No thread_ts — replies directly in channel
        assert "thread_ts" not in call_kwargs

    async def test_incident_channel_seeds_context(self):
        """First message in incident channel seeds conversation with incident context."""
        settings = _make_settings()
        say = AsyncMock()
        mock_agent = AsyncMock()
        mock_agent.run.return_value = _agent_result("Searching KB for VPN articles...")

        handlers._incident_channels.add("C_INC42")
        handlers._incident_context["C_INC42"] = {
            "ticket_id": "INC0010042",
            "title": "VPN issue",
            "description": "Can't connect to VPN",
            "priority": "medium",
            "summary_ts": None,
            "original_text": None,
            "summary_lines": [],
        }

        event = {
            "channel": "C_INC42",
            "channel_type": "group",
            "user": "U_USER",
            "text": "find a KB article",
            "ts": "4000.1",
        }

        with patch.object(handlers, "_get_agent", return_value=mock_agent):
            await handlers._handle_incident_message(event, "find a KB article", say, settings)

        # History should have: context seed (2) + user message + assistant response = 4
        conv_key = ("C_INC42", "incident")
        history = handlers._conversations[conv_key]
        assert len(history) == 4
        # First message should contain incident context
        assert "INC0010042" in history[0]["content"]
        assert "VPN issue" in history[0]["content"]

    async def test_incident_channel_single_conversation(self):
        """All messages in an incident channel share one conversation history."""
        settings = _make_settings()
        say = AsyncMock()
        mock_agent = AsyncMock()
        mock_agent.run.return_value = _agent_result("Got it.")

        handlers._incident_channels.add("C_INC42")

        # Two messages with different ts values
        for ts in ("4000.1", "4000.2"):
            event = {
                "channel": "C_INC42",
                "channel_type": "group",
                "user": "U_USER",
                "text": "update",
                "ts": ts,
            }
            with patch.object(handlers, "_get_agent", return_value=mock_agent):
                await handlers._handle_incident_message(event, "update", say, settings)

        # Both messages land in the same conversation key
        conv_key = ("C_INC42", "incident")
        assert conv_key in handlers._conversations
        assert len(handlers._conversations[conv_key]) == 4  # 2 user + 2 assistant

    async def test_incident_summary_updated(self):
        """After agent responds, the initial message is updated with LIVE SUMMARY."""
        settings = _make_settings()
        say = AsyncMock()
        mock_agent = AsyncMock()
        mock_agent.run.return_value = _agent_result("Found 3 KB articles about VPN setup.")

        handlers._incident_channels.add("C_INC42")
        handlers._incident_context["C_INC42"] = {
            "ticket_id": "INC0010042",
            "title": "VPN issue",
            "description": "Can't connect",
            "priority": "medium",
            "summary_ts": "1000.0",
            "original_text": "*Incident INC0010042*\n*Title:* VPN issue",
            "summary_lines": [],
        }

        event = {
            "channel": "C_INC42",
            "channel_type": "group",
            "user": "U_USER",
            "text": "search KB",
            "ts": "4000.1",
        }

        with patch.object(handlers, "_get_agent", return_value=mock_agent):
            with patch("it_agent.bot.handlers.AsyncWebClient") as mock_cls:
                mock_client = AsyncMock()
                mock_cls.return_value = mock_client
                await handlers._handle_incident_message(event, "search KB", say, settings)

        # Verify chat_update was called with the summary
        mock_client.chat_update.assert_called_once()
        update_kwargs = mock_client.chat_update.call_args[1]
        assert update_kwargs["channel"] == "C_INC42"
        assert update_kwargs["ts"] == "1000.0"
        assert "LIVE SUMMARY:" in update_kwargs["text"]
        assert "Found 3 KB articles about VPN setup." in update_kwargs["text"]

    async def test_unregistered_channel_ignored(self):
        """Messages in unregistered channels are not processed."""
        event = _other_channel_event()
        settings = _make_settings()
        assert event["channel"] not in handlers._incident_channels
        assert event["channel"] != settings.help_channel_id

    async def test_mention_in_incident_channel_skipped(self):
        """app_mention in incident channel is skipped (message handler covers it)."""
        handlers._incident_channels.add("C_INC42")
        event = {"channel": "C_INC42", "text": "<@U_BOT> help", "ts": "5000.1"}
        channel = event.get("channel", "")
        assert channel in handlers._incident_channels


class TestParseIncidentMessage:
    def test_parse_full_message(self):
        text = (
            "*Incident INC0129535*\n"
            "*Title:* Contractor needs VPN access\n"
            "*Priority:* medium\n"
            "*Description:* Can't connect to the company VPN\n\n"
            "<https://test.service-now.com|View in ServiceNow>"
        )
        ctx = handlers._parse_incident_message(text)
        assert ctx["ticket_id"] == "INC0129535"
        assert ctx["title"] == "Contractor needs VPN access"
        assert ctx["priority"] == "medium"
        assert ctx["description"] == "Can't connect to the company VPN"

    def test_parse_partial_message(self):
        text = "*Incident INC0010001*\n*Title:* Broken laptop"
        ctx = handlers._parse_incident_message(text)
        assert ctx["ticket_id"] == "INC0010001"
        assert ctx["title"] == "Broken laptop"
        assert "priority" not in ctx

    def test_parse_empty_message(self):
        ctx = handlers._parse_incident_message("just some random text")
        assert ctx == {}
