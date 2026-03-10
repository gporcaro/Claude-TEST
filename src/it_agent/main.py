import asyncio
import logging

from google import genai
from qdrant_client import QdrantClient

from it_agent.bot.app import create_app, start_app
from it_agent.bot import events
from it_agent.bot.handlers import discover_incident_channels, init_shared_clients, load_ai_context_articles, recover_active_refinements, resolve_debug_channel, start_approval_timeout_loop, start_auto_close_loop, start_collaborative_review_timeout_loop, start_recommendation_timeout_loop
from it_agent.config import get_settings
from it_agent.db import init_db
from it_agent.kb.indexer import index_knowledge_base

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(_start(settings))


async def _start(settings) -> None:
    events.configure(settings.dashboard_url)
    await init_db(settings.db_path)

    init_shared_clients(settings)
    await _ensure_kb_indexed(settings)
    await discover_incident_channels(settings)
    await load_ai_context_articles(settings)
    await resolve_debug_channel(settings)
    await recover_active_refinements(settings)

    from it_agent.bot.handlers import _incident_channels, _resolved_pending_close
    await events.emit("bot_startup", {
        "channels_discovered": len(_incident_channels),
        "pending_auto_close": len(_resolved_pending_close),
    })

    app = create_app(settings)

    # Start the auto-close background loop (closes resolved tickets after 48h)
    asyncio.create_task(start_auto_close_loop(settings))

    # Start the approval timeout loop (auto-denies stale article approvals after 30min)
    asyncio.create_task(start_approval_timeout_loop(settings))

    # Start the recommendation approval timeout loop (auto-denies after 30min)
    asyncio.create_task(start_recommendation_timeout_loop(settings))

    # Start the collaborative review timeout loop (falls back to standard approval)
    asyncio.create_task(start_collaborative_review_timeout_loop(settings))

    await start_app(app, settings)


async def _ensure_kb_indexed(settings) -> None:
    """Auto-index KB on startup if the Qdrant collection is empty or missing."""
    try:
        qclient = QdrantClient(url=settings.qdrant_url)
        existing = [c.name for c in qclient.get_collections().collections]

        if settings.qdrant_collection in existing:
            info = qclient.get_collection(settings.qdrant_collection)
            if info.points_count and info.points_count > 0:
                logger.info(
                    "KB already indexed (%d vectors), skipping auto-index",
                    info.points_count,
                )
                return

        logger.info("KB collection empty or missing, starting auto-index...")
        client = genai.Client(api_key=settings.gemini_api_key)
        summary = await index_knowledge_base(settings, client, recreate=False)
        logger.info(
            "Auto-index complete: %d articles, %d chunks",
            summary["articles"],
            summary["chunks"],
        )
    except Exception:
        logger.warning("Auto-index failed — bot will start without KB search", exc_info=True)


if __name__ == "__main__":
    main()
