"""CLI entry point for `it-agent-index` — full KB re-index."""

from __future__ import annotations

import asyncio
import logging
import sys

from google import genai

from it_agent.config import get_settings
from it_agent.kb.indexer import index_knowledge_base


async def _run() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    print("Starting full KB re-index...")
    client = genai.Client(api_key=settings.gemini_api_key)

    try:
        summary = await index_knowledge_base(settings, client, recreate=True)
    except Exception as e:
        print(f"Indexing failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nIndexing complete:")
    print(f"  Articles indexed: {summary['articles']}")
    print(f"  Chunks created:   {summary['chunks']}")
    print(f"  Sources:          {', '.join(summary['sources']) or 'none'}")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
