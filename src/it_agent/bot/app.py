from __future__ import annotations

import logging

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

from it_agent.config import Settings

logger = logging.getLogger(__name__)


def create_app(settings: Settings) -> AsyncApp:
    """Create and configure the Slack AsyncApp."""
    app = AsyncApp(token=settings.slack_bot_token)

    from it_agent.bot.handlers import register_handlers

    register_handlers(app, settings)

    return app


async def start_app(app: AsyncApp, settings: Settings) -> None:
    """Start the Slack app in Socket Mode."""
    handler = AsyncSocketModeHandler(app, settings.slack_app_token)
    logger.info("Starting IT Support Agent in Socket Mode...")
    await handler.start_async()
