"""CLI entry point for `it-agent-index-public` — index public vendor articles."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from google import genai

from it_agent.config import get_settings
from it_agent.db import init_db
from it_agent.kb.public_indexer import index_public_articles


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    await init_db(settings.db_path)

    urls: list[dict] | None = None

    if args.url:
        from urllib.parse import urlparse
        domain = urlparse(args.url).hostname or ""
        urls = [{"url": args.url, "source_domain": domain}]
        print(f"Indexing single URL: {args.url}")
    elif args.urls_file:
        from urllib.parse import urlparse
        urls = []
        with open(args.urls_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    domain = urlparse(line).hostname or ""
                    urls.append({"url": line, "source_domain": domain})
        print(f"Indexing {len(urls)} URLs from {args.urls_file}")
    else:
        print("Crawling vendor sitemaps...")

    client = genai.Client(api_key=settings.gemini_api_key)

    try:
        summary = await index_public_articles(
            settings, client, urls=urls, recreate=args.recreate,
        )
    except Exception as e:
        print(f"Indexing failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nIndexing complete:")
    print(f"  Articles indexed: {summary['articles']}")
    print(f"  Chunks created:   {summary['chunks']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index public vendor support articles")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sitemap", action="store_true", default=True,
                       help="Crawl vendor sitemaps (default)")
    group.add_argument("--url", type=str, help="Index a single URL")
    group.add_argument("--urls-file", type=str, help="Read URLs from a text file (one per line)")
    parser.add_argument("--recreate", action="store_true",
                        help="Recreate the Qdrant collection from scratch")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
