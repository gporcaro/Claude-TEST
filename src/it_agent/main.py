import asyncio
import logging

from it_agent.bot.app import create_app, start_app
from it_agent.config import get_settings
from it_agent.db.database import init_db


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(_start(settings))


async def _start(settings) -> None:
    await init_db(settings.db_path)
    app = create_app(settings)
    await start_app(app, settings)


if __name__ == "__main__":
    main()
