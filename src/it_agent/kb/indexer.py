"""KB indexing pipeline: fetch articles -> chunk -> embed -> upsert to Qdrant."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from it_agent.config import Settings
from it_agent.kb.embeddings import embed_texts
from it_agent.servicenow.client import ServiceNowClient

logger = logging.getLogger(__name__)

_VECTOR_DIM = 768
_CHUNK_WORDS = 500
_OVERLAP_WORDS = 50
_UPSERT_BATCH = 100
_DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def _strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", "", text)


def _chunk_text(text: str) -> list[str]:
    """Split text into ~500-word chunks with 50-word overlap."""
    words = text.split()
    if len(words) <= _CHUNK_WORDS:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + _CHUNK_WORDS
        chunks.append(" ".join(words[start:end]))
        start = end - _OVERLAP_WORDS
    return chunks


def _point_id(article_id: str, chunk_index: int) -> str:
    """Deterministic point ID from article_id + chunk_index."""
    raw = f"{article_id}_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()  # noqa: S324


def _ensure_collection(qclient: QdrantClient, collection: str, recreate: bool) -> None:
    """Create the Qdrant collection if it doesn't exist (or recreate)."""
    if recreate:
        qclient.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=_VECTOR_DIM, distance=Distance.COSINE),
        )
        return

    existing = [c.name for c in qclient.get_collections().collections]
    if collection not in existing:
        qclient.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=_VECTOR_DIM, distance=Distance.COSINE),
        )


async def _fetch_servicenow_articles(settings: Settings) -> list[dict]:
    """Fetch all published KB articles from ServiceNow."""
    client = ServiceNowClient(
        settings.sn_instance_url, settings.sn_username, settings.sn_password
    )
    try:
        return await client.list_kb_articles(limit=500)
    finally:
        await client.close()


def _load_local_docs() -> list[dict]:
    """Load all *.md files from the docs/ directory."""
    articles: list[dict] = []
    if not _DOCS_DIR.is_dir():
        logger.info("No docs/ directory found at %s", _DOCS_DIR)
        return articles

    for md_file in sorted(_DOCS_DIR.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        # Extract title from first # heading, fall back to filename
        title = md_file.stem.replace("-", " ").title()
        for line in content.splitlines():
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break

        articles.append({
            "id": f"local_{md_file.stem}",
            "title": title,
            "content": content,
            "category": "Local Docs",
            "source": "local",
        })
    return articles


def _prepare_chunks(articles: list[dict]) -> list[dict]:
    """Turn articles into chunks ready for embedding."""
    all_chunks: list[dict] = []
    for article in articles:
        raw_content = article.get("content", "")
        source = article.get("source", "servicenow")

        # Strip HTML for ServiceNow articles
        if source != "local":
            raw_content = _strip_html(raw_content)

        chunks = _chunk_text(raw_content)
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "article_id": article["id"],
                "title": article.get("title", ""),
                "content": chunk_text[:1000],
                "source": source,
                "category": article.get("category", ""),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "full_text": chunk_text,
            })
    return all_chunks


async def index_knowledge_base(
    settings: Settings,
    genai_client,
    recreate: bool = False,
) -> dict:
    """Full indexing pipeline: fetch -> chunk -> embed -> upsert.

    Returns summary dict with counts.
    """
    # 1. Fetch articles from both sources
    sn_articles: list[dict] = []
    try:
        sn_articles = await _fetch_servicenow_articles(settings)
        for a in sn_articles:
            a.setdefault("source", "servicenow")
        logger.info("Fetched %d articles from ServiceNow", len(sn_articles))
    except Exception:
        logger.warning("Failed to fetch ServiceNow articles, continuing with local docs only")

    local_articles = _load_local_docs()
    logger.info("Loaded %d local doc(s)", len(local_articles))

    all_articles = sn_articles + local_articles
    if not all_articles:
        return {"articles": 0, "chunks": 0, "sources": []}

    # 2. Chunk
    chunks = _prepare_chunks(all_articles)
    logger.info("Created %d chunks from %d articles", len(chunks), len(all_articles))

    # 3. Embed
    texts_to_embed = [c["full_text"] for c in chunks]
    vectors = await embed_texts(genai_client, texts_to_embed)

    # 4. Upsert to Qdrant
    qclient = QdrantClient(url=settings.qdrant_url)
    _ensure_collection(qclient, settings.qdrant_collection, recreate=recreate)

    points: list[PointStruct] = []
    for chunk, vector in zip(chunks, vectors):
        pid = _point_id(chunk["article_id"], chunk["chunk_index"])
        payload = {
            "article_id": chunk["article_id"],
            "title": chunk["title"],
            "content": chunk["content"],
            "source": chunk["source"],
            "category": chunk["category"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
        }
        points.append(PointStruct(id=pid, vector=vector, payload=payload))

    # Batch upsert
    for i in range(0, len(points), _UPSERT_BATCH):
        batch = points[i : i + _UPSERT_BATCH]
        qclient.upsert(collection_name=settings.qdrant_collection, points=batch)

    sources = sorted({a.get("source", "unknown") for a in all_articles})
    logger.info(
        "Indexed %d chunks from %d articles (sources: %s)",
        len(chunks),
        len(all_articles),
        ", ".join(sources),
    )

    return {
        "articles": len(all_articles),
        "chunks": len(chunks),
        "sources": sources,
    }
