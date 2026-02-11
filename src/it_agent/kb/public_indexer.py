"""Public article indexer: sitemap crawl, content fetch, embed, upsert to Qdrant."""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from it_agent import db
from it_agent.config import Settings
from it_agent.kb.embeddings import embed_texts
from it_agent.kb.indexer import _chunk_text, _ensure_collection, _point_id, _strip_html

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 100

# Default vendor sitemap URLs
_DEFAULT_SITEMAPS = {
    "support.apple.com": "https://support.apple.com/sitemap.xml",
    "support.microsoft.com": "https://support.microsoft.com/sitemap.xml",
    "www.dell.com": "https://www.dell.com/support/sitemap.xml",
}


async def crawl_vendor_sitemaps(
    domains: dict[str, str] | None = None,
    max_per_domain: int = 100,
) -> list[dict]:
    """Fetch sitemap.xml for each vendor and return article URLs.

    Returns a list of ``{url, source_domain}`` dicts.
    """
    sitemaps = domains or _DEFAULT_SITEMAPS
    all_urls: list[dict] = []

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for domain, sitemap_url in sitemaps.items():
            try:
                resp = await client.get(sitemap_url)
                resp.raise_for_status()
                urls = _parse_sitemap(resp.text, domain, max_per_domain)
                all_urls.extend(urls)
                logger.info("Crawled %d URLs from %s", len(urls), domain)
            except Exception:
                logger.warning("Failed to crawl sitemap for %s", domain, exc_info=True)

    return all_urls


def _parse_sitemap(xml_text: str, domain: str, max_urls: int) -> list[dict]:
    """Parse a sitemap XML and extract article URLs."""
    urls: list[dict] = []
    try:
        root = ElementTree.fromstring(xml_text)  # noqa: S314
        # Handle both sitemap index and urlset
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for url_elem in root.findall(".//sm:url/sm:loc", ns):
            if url_elem.text:
                urls.append({"url": url_elem.text.strip(), "source_domain": domain})
                if len(urls) >= max_urls:
                    break
        # Fallback: try without namespace
        if not urls:
            for url_elem in root.iter():
                if url_elem.tag.endswith("loc") and url_elem.text:
                    urls.append({"url": url_elem.text.strip(), "source_domain": domain})
                    if len(urls) >= max_urls:
                        break
    except ElementTree.ParseError:
        logger.warning("Failed to parse sitemap XML for %s", domain)
    return urls


async def fetch_article_content(url: str) -> dict | None:
    """Fetch a URL and extract article content.

    Returns ``{url, title, content, snippet}`` or *None* on failure.
    Uses BeautifulSoup with lxml parser for content extraction.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 not installed, falling back to basic HTML stripping")
        return await _fetch_article_basic(url)

    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "IT-Agent-Bot/1.0"})
            resp.raise_for_status()
            html = resp.text
    except Exception:
        logger.warning("Failed to fetch %s", url, exc_info=True)
        return None

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # Remove nav, footer, script, style
    for tag in soup.find_all(["nav", "footer", "script", "style", "header", "aside"]):
        tag.decompose()

    # Extract title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

    # Extract main content
    main = soup.find("main") or soup.find("article") or soup.find("div", {"role": "main"})
    if main:
        content = main.get_text(separator="\n", strip=True)
    else:
        content = soup.get_text(separator="\n", strip=True)

    # Clean up excessive whitespace
    content = re.sub(r"\n{3,}", "\n\n", content).strip()

    snippet = content[:200] if content else ""

    return {"url": url, "title": title or url, "content": content, "snippet": snippet}


async def _fetch_article_basic(url: str) -> dict | None:
    """Basic fallback when beautifulsoup4 is not installed."""
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "IT-Agent-Bot/1.0"})
            resp.raise_for_status()
            html = resp.text
    except Exception:
        logger.warning("Failed to fetch %s", url, exc_info=True)
        return None

    # Extract title from <title> tag
    title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else url

    content = _strip_html(html).strip()
    content = re.sub(r"\n{3,}", "\n\n", content)
    snippet = content[:200]

    return {"url": url, "title": title, "content": content, "snippet": snippet}


async def index_public_articles(
    settings: Settings,
    genai_client,
    urls: list[dict] | None = None,
    recreate: bool = False,
) -> dict:
    """Full pipeline: fetch URLs -> store in SQLite -> chunk -> embed -> upsert to Qdrant.

    *urls* should be a list of ``{url, source_domain}`` dicts.
    If *None*, crawls vendor sitemaps.

    Returns summary dict with counts.
    """
    if urls is None:
        urls = await crawl_vendor_sitemaps()

    if not urls:
        return {"articles": 0, "chunks": 0}

    articles: list[dict] = []
    for entry in urls:
        url = entry["url"]
        domain = entry.get("source_domain") or urlparse(url).hostname or ""

        # Check if already in DB
        existing = await db.get_public_article_by_url(url)
        if existing:
            articles.append(existing)
            continue

        # Fetch content
        fetched = await fetch_article_content(url)
        if not fetched or not fetched.get("content"):
            continue

        # Store in SQLite as curated
        article_id = await db.create_public_article(
            url=url,
            title=fetched["title"],
            content=fetched["content"],
            snippet=fetched["snippet"],
            source_domain=domain,
            status="curated",
        )
        article = await db.get_public_article(article_id)
        if article:
            articles.append(article)

    if not articles:
        return {"articles": 0, "chunks": 0}

    # Chunk all articles
    all_chunks: list[dict] = []
    for article in articles:
        content = article.get("content", "").strip()
        if not content:
            continue
        chunks = _chunk_text(content)
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
            all_chunks.append({
                "article_id": article["id"],
                "url": article["url"],
                "title": article["title"],
                "source_domain": article["source_domain"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "full_text": chunk_text,
            })

    if not all_chunks:
        return {"articles": len(articles), "chunks": 0}

    # Embed
    texts_to_embed = [c["full_text"] for c in all_chunks]
    vectors = await embed_texts(genai_client, texts_to_embed)

    # Upsert to Qdrant
    qclient = QdrantClient(url=settings.qdrant_url)
    _ensure_collection(qclient, settings.qdrant_public_collection, recreate=recreate)

    points: list[PointStruct] = []
    for chunk, vector in zip(all_chunks, vectors):
        pid = _point_id(f"pub_{chunk['article_id']}", chunk["chunk_index"])
        payload = {
            "article_id": chunk["article_id"],
            "url": chunk["url"],
            "title": chunk["title"],
            "content": chunk["full_text"][:1000],
            "source_domain": chunk["source_domain"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
        }
        points.append(PointStruct(id=pid, vector=vector, payload=payload))

    for i in range(0, len(points), _UPSERT_BATCH):
        batch = points[i : i + _UPSERT_BATCH]
        qclient.upsert(collection_name=settings.qdrant_public_collection, points=batch)

    # Mark articles as indexed
    for article in articles:
        await db.update_public_article(article["id"], qdrant_indexed=1)

    logger.info(
        "Indexed %d public article chunks from %d articles",
        len(all_chunks), len(articles),
    )

    return {"articles": len(articles), "chunks": len(all_chunks)}


async def index_single_article(
    article_id: int,
    settings: Settings,
    genai_client,
) -> bool:
    """Index a single approved article into Qdrant. Called on IT approval.

    Returns *True* on success.
    """
    article = await db.get_public_article(article_id)
    if not article:
        logger.warning("Article %d not found for indexing", article_id)
        return False

    # Fetch content if not already stored
    if not article.get("content"):
        fetched = await fetch_article_content(article["url"])
        if not fetched or not fetched.get("content"):
            logger.warning("Could not fetch content for article %d", article_id)
            return False
        await db.update_public_article(
            article_id,
            content=fetched["content"],
            snippet=fetched["snippet"],
            title=fetched.get("title") or article["title"],
        )
        article = await db.get_public_article(article_id)
        if not article:
            return False

    content = article.get("content", "").strip()
    if not content:
        return False

    # Chunk and embed
    chunks = _chunk_text(content)
    texts = [c for c in chunks if c.strip()]
    if not texts:
        return False

    vectors = await embed_texts(genai_client, texts)

    # Upsert to Qdrant
    qclient = QdrantClient(url=settings.qdrant_url)
    _ensure_collection(qclient, settings.qdrant_public_collection, recreate=False)

    points: list[PointStruct] = []
    for i, (chunk_text, vector) in enumerate(zip(texts, vectors)):
        pid = _point_id(f"pub_{article_id}", i)
        payload = {
            "article_id": article_id,
            "url": article["url"],
            "title": article["title"],
            "content": chunk_text[:1000],
            "source_domain": article["source_domain"],
            "chunk_index": i,
            "total_chunks": len(texts),
        }
        points.append(PointStruct(id=pid, vector=vector, payload=payload))

    qclient.upsert(collection_name=settings.qdrant_public_collection, points=points)
    await db.update_public_article(article_id, qdrant_indexed=1)
    logger.info("Indexed article %d (%s) into Qdrant", article_id, article["url"])
    return True
