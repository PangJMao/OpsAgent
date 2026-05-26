from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ops_agent.services.document_processing.chunkers import ChunkerRouter
from ops_agent.services.document_processing.cleaners import DocumentCleaner
from ops_agent.services.document_processing.loaders import DocumentLoader
from ops_agent.services.document_processing.metadata import MetadataBuilder
from ops_agent.services.document_processing.schema import IngestedChunk
from ops_agent.services.document_processing.vector_store import VectorStore, create_document_vector_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    source_count: int
    chunk_count: int
    skipped_count: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    chunks: list[IngestedChunk] = field(default_factory=list)


class IngestionPipeline:
    def __init__(
        self,
        loader: DocumentLoader | None = None,
        cleaner: DocumentCleaner | None = None,
        chunker: ChunkerRouter | None = None,
        metadata_builder: MetadataBuilder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.loader = loader or DocumentLoader()
        self.cleaner = cleaner or DocumentCleaner()
        self.chunker = chunker or ChunkerRouter()
        self.metadata_builder = metadata_builder or MetadataBuilder()
        self.vector_store = vector_store or create_document_vector_store()
        self._source_hashes: dict[str, str] = {}

    def ingest_path(self, path: Path, write: bool = True) -> IngestionResult:
        files = self.loader.iter_files(path)
        all_chunks: list[IngestedChunk] = []
        failed: dict[str, str] = {}
        skipped = 0
        for file_path in files:
            try:
                chunks = self._prepare_file(file_path)
                content_signature = "|".join(sorted(chunk.metadata["content_hash"] for chunk in chunks))
                source = str(file_path)
                if self._source_hashes.get(source) == content_signature:
                    skipped += 1
                    continue
                self._source_hashes[source] = content_signature
                all_chunks.extend(chunks)
                logger.info("Prepared %s chunks from %s", len(chunks), file_path)
            except Exception as exc:  # pragma: no cover - defensive logging path
                failed[str(file_path)] = str(exc)
                logger.exception("Failed to ingest %s", file_path)

        if write and all_chunks:
            self.vector_store.upsert_chunks(_dedupe_by_hash(all_chunks))
        return IngestionResult(
            source_count=len(files),
            chunk_count=len(all_chunks),
            skipped_count=skipped,
            failed=failed,
            chunks=all_chunks,
        )

    def delete_by_source(self, source: str) -> None:
        self.vector_store.delete_by_source(source)
        self._source_hashes.pop(source, None)

    def _prepare_file(self, path: Path) -> list[IngestedChunk]:
        parsed = self.loader.load(path)
        cleaned = self.cleaner.clean(parsed)
        chunks = self.chunker.chunk(cleaned)
        return self.metadata_builder.build(chunks)


def _dedupe_by_hash(chunks: list[IngestedChunk]) -> list[IngestedChunk]:
    seen: set[str] = set()
    deduped: list[IngestedChunk] = []
    for chunk in chunks:
        content_hash = str(chunk.metadata["content_hash"])
        if content_hash in seen:
            continue
        seen.add(content_hash)
        deduped.append(chunk)
    return deduped
