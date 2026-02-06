"""Index markdown docs into ChromaDB."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from it_agent.knowledge.store import get_client, get_collection

logger = logging.getLogger(__name__)


def _split_markdown(text: str, source: str) -> list[dict]:
    """Split a markdown file into chunks by headers."""
    chunks = []
    # Split on ## or # headers
    sections = re.split(r"\n(?=#{1,2}\s)", text)

    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        # Extract title from first line if it's a header
        lines = section.split("\n", 1)
        title = lines[0].lstrip("#").strip() if lines[0].startswith("#") else ""

        chunks.append(
            {
                "id": f"{source}::chunk_{i}",
                "content": section,
                "metadata": {
                    "source": source,
                    "title": title,
                    "chunk_index": i,
                },
            }
        )
    return chunks


def index_docs(docs_path: Path, chroma_path: Path) -> int:
    """Index all markdown files from docs_path into ChromaDB. Returns count of chunks indexed."""
    client = get_client(chroma_path)
    # Delete and recreate to do a fresh index
    try:
        client.delete_collection("it_knowledge_base")
    except Exception:
        pass
    collection = get_collection(client)

    all_chunks = []
    md_files = list(docs_path.glob("**/*.md"))
    logger.info("Found %d markdown files in %s", len(md_files), docs_path)

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        source = md_file.stem
        chunks = _split_markdown(text, source)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("No chunks to index")
        return 0

    # Batch add to ChromaDB
    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["content"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks],
    )

    logger.info("Indexed %d chunks from %d files", len(all_chunks), len(md_files))
    return len(all_chunks)


def main() -> None:
    """CLI entry point for indexing docs."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Index IT knowledge base docs")
    parser.add_argument(
        "--docs", type=Path, default=Path("src/it_agent/docs"), help="Path to markdown docs"
    )
    parser.add_argument(
        "--chroma", type=Path, default=Path("chroma_data"), help="ChromaDB storage path"
    )
    args = parser.parse_args()

    count = index_docs(args.docs, args.chroma)
    print(f"Indexed {count} chunks")


if __name__ == "__main__":
    main()
