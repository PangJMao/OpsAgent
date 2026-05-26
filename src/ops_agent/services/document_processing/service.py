from __future__ import annotations

from pathlib import Path

from ops_agent.models import Chunk, Document, NormalizedDocument
from ops_agent.services.document_processing.loaders import DocumentLoader
from ops_agent.services.document_processing.pipeline import IngestionPipeline, IngestionResult


class DocumentProcessingService:
    """External facade for document parsing, chunking, metadata generation and ingestion."""

    def __init__(self, pipeline: IngestionPipeline | None = None) -> None:
        self.pipeline = pipeline or IngestionPipeline()
        self.loader = self.pipeline.loader

    def ingest(self, path: Path, write: bool = True) -> IngestionResult:
        return self.pipeline.ingest_path(path, write=write)

    def load_as_document(self, path: Path) -> Document:
        parsed = self.loader.load(path)
        markdown = self.to_markdown(path)
        document_id = self.pipeline.metadata_builder.build(self.pipeline.chunker.chunk(self.pipeline.cleaner.clean(parsed))[:1])
        stable_id = document_id[0].document_id if document_id else path.stem
        return Document(
            document_id=stable_id,
            title=parsed.title,
            source_path=parsed.source,
            content=markdown.markdown,
            metadata={**parsed.metadata, "source_format": parsed.file_type, "normalizer": parsed.parser, "normalized_format": "markdown"},
        )

    def to_markdown(self, path: Path) -> NormalizedDocument:
        parsed = self.loader.load(path)
        lines = [f"# {parsed.title}"]
        for block in parsed.blocks:
            if block.page:
                lines.append(f"\n\n## Page {block.page}\n\n{block.text}")
            elif block.sheet_name and block.block_type == "table":
                lines.append(f"\n\n## {block.sheet_name}\n\n{block.text}")
            else:
                lines.append(f"\n\n{block.text}")
        return NormalizedDocument(
            title=parsed.title,
            markdown="".join(lines).strip(),
            metadata={**parsed.metadata, "source_format": parsed.file_type, "normalizer": parsed.parser},
        )

    def chunk_document(self, document: Document) -> list[Chunk]:
        from ops_agent.services.document_service import chunk_document

        return chunk_document(document)
