"""ChromaDB query wrapper for knowledge base search."""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "it_knowledge_base"


def get_client(chroma_path: Path) -> chromadb.ClientAPI:
    """Get a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=str(chroma_path))


def get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Get or create the knowledge base collection."""
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


async def search(chroma_path: Path, query: str, n_results: int = 3) -> list[dict]:
    """Search the knowledge base and return matching documents."""
    import asyncio

    def _search():
        client = get_client(chroma_path)
        collection = get_collection(client)

        if collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, 10),
        )

        docs = []
        for i in range(len(results["ids"][0])):
            doc = {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            }
            if results.get("metadatas") and results["metadatas"][0][i]:
                doc["metadata"] = results["metadatas"][0][i]
            docs.append(doc)
        return docs

    return await asyncio.get_event_loop().run_in_executor(None, _search)
