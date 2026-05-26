from __future__ import annotations

import logging
from typing import Protocol

from ops_agent.config import settings
from ops_agent.models import Chunk
from ops_agent.services.document_processing.schema import IngestedChunk
from ops_agent.services.vector_store import LocalVectorStore, PgVectorStore

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    def upsert_chunks(self, chunks: list[IngestedChunk]) -> None:
        ...

    def delete_by_source(self, source: str) -> None:
        ...

    def count(self) -> int:
        ...


class PgDocumentVectorStore:
    """Adapter for the project's existing PostgreSQL + pgvector knowledge_chunks table."""

    def __init__(self, store: PgVectorStore | None = None) -> None:
        self.store = store or PgVectorStore()

    def upsert_chunks(self, chunks: list[IngestedChunk]) -> None:
        self.store.upsert_chunks(_to_domain_chunks(chunks))

    def delete_by_source(self, source: str) -> None:
        self.store.mark_deleted_by_source(source)

    def count(self) -> int:
        return self.store.count()


class LocalDocumentVectorStore:
    """Adapter for the existing SQLite vector store; useful for tests and local MVP runs."""

    def __init__(self, store: LocalVectorStore | None = None) -> None:
        self.store = store or LocalVectorStore()

    def upsert_chunks(self, chunks: list[IngestedChunk]) -> None:
        self.store.upsert_chunks(_to_domain_chunks(chunks))

    def delete_by_source(self, source: str) -> None:
        self.store.mark_deleted_by_source(source)

    def count(self) -> int:
        return self.store.count()


def create_document_vector_store() -> VectorStore:
    provider = getattr(settings, "document_vector_provider", settings.vector_provider).lower()
    if provider in {"pgvector", "pgsql", "postgres", "postgresql"}:
        return PgDocumentVectorStore()
    return LocalDocumentVectorStore()


def _to_domain_chunks(chunks: list[IngestedChunk]) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            title=chunk.title,
            text=chunk.text,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            metadata=chunk.metadata,
        )
        for chunk in chunks
    ]
