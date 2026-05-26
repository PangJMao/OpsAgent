from __future__ import annotations

import re
from abc import ABC, abstractmethod

from ops_agent.config import settings
from ops_agent.services.document_processing.schema import DocumentChunk, ParsedBlock, ParsedDocument

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class DocumentChunker(ABC):
    strategy = "base"

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        ...


class MarkdownHeaderChunker(DocumentChunker):
    strategy = "markdown_header"

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        text = "\n\n".join(block.text for block in document.blocks)
        sections = _markdown_sections(text)
        if not sections:
            return RecursiveTextSplitter().chunk(document)
        chunks: list[DocumentChunk] = []
        for section_text, start, end, heading_path, level in sections:
            if len(section_text) <= settings.chunk_size:
                chunks.append(_chunk(document, section_text, start, end, len(chunks), self.strategy, heading_path, {"section_level": level}))
            else:
                chunks.extend(_split_text(document, section_text, start, heading_path, len(chunks), f"{self.strategy}_recursive"))
        return chunks


class RecursiveTextSplitter(DocumentChunker):
    strategy = "recursive_text"

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        text = "\n\n".join(block.text for block in document.blocks)
        return _split_text(document, text, 0, [], 0, self.strategy)


class TableAwareChunker(DocumentChunker):
    strategy = "table_aware"

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for block in document.blocks:
            if block.block_type in {"table", "table_row"}:
                chunks.append(_block_chunk(document, block, len(chunks), self.strategy))
            else:
                pseudo = ParsedDocument(document.source, document.file_name, document.file_type, document.title, document.parser, [block], document.metadata)
                chunks.extend(_split_text(pseudo, block.text, 0, block.heading_path, len(chunks), "table_context_recursive"))
        return chunks


class SlideChunker(DocumentChunker):
    strategy = "slide"

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        return [_block_chunk(document, block, index, self.strategy) for index, block in enumerate(document.blocks)]


class PDFLayoutChunker(DocumentChunker):
    strategy = "pdf_layout"

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for block in document.blocks:
            pseudo = ParsedDocument(document.source, document.file_name, document.file_type, document.title, document.parser, [block], document.metadata)
            if len(block.text) <= settings.chunk_size:
                chunks.append(_block_chunk(document, block, len(chunks), self.strategy))
            else:
                page_chunks = _split_text(pseudo, block.text, 0, block.heading_path, len(chunks), "pdf_layout_recursive")
                chunks.extend(page_chunks)
        return chunks


class ChunkerRouter:
    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        if document.file_type in {"markdown", "docx", "html"}:
            return MarkdownHeaderChunker().chunk(document)
        if document.file_type in {"xlsx", "csv"}:
            return TableAwareChunker().chunk(document)
        if document.file_type == "pdf":
            return PDFLayoutChunker().chunk(document)
        if document.file_type == "pptx":
            return SlideChunker().chunk(document)
        return RecursiveTextSplitter().chunk(document)


def _markdown_sections(text: str) -> list[tuple[str, int, int, list[str], int]]:
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return []
    sections = []
    stack: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        stack = [(item_level, item_title) for item_level, item_title in stack if item_level < level]
        stack.append((level, title))
        sections.append((text[start:end].strip(), start, end, [item_title for _, item_title in stack], level))
    return sections


def _split_text(document: ParsedDocument, text: str, base_start: int, heading_path: list[str], start_index: int, strategy: str) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(text):
        end = min(start + settings.chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            boundary = max(window.rfind("\n\n"), window.rfind("。"), window.rfind("."), window.rfind(";"), window.rfind("；"))
            if boundary > settings.chunk_size * 0.55:
                end = start + boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(_chunk(document, piece, base_start + start, base_start + end, start_index + len(chunks), strategy, heading_path, {}))
        if end >= len(text):
            break
        start = max(0, end - settings.chunk_overlap)
    return chunks


def _block_chunk(document: ParsedDocument, block: ParsedBlock, index: int, strategy: str) -> DocumentChunk:
    return DocumentChunk(
        text=_with_heading_context(block.text, block.heading_path),
        source=document.source,
        file_name=document.file_name,
        file_type=document.file_type,
        page=block.page,
        sheet_name=block.sheet_name,
        row_index=block.row_index,
        slide_number=block.slide_number,
        section=block.section,
        heading_path=block.heading_path,
        start_char=0,
        end_char=len(block.text),
        chunk_index=index,
        strategy=strategy,
        parser=document.parser,
        metadata=block.metadata,
    )


def _chunk(
    document: ParsedDocument,
    text: str,
    start: int,
    end: int,
    index: int,
    strategy: str,
    heading_path: list[str],
    metadata: dict[str, object],
) -> DocumentChunk:
    return DocumentChunk(
        text=_with_heading_context(text, heading_path),
        source=document.source,
        file_name=document.file_name,
        file_type=document.file_type,
        section=heading_path[-1] if heading_path else None,
        heading_path=heading_path,
        start_char=start,
        end_char=end,
        chunk_index=index,
        strategy=strategy,
        parser=document.parser,
        metadata=metadata,
    )


def _with_heading_context(text: str, heading_path: list[str]) -> str:
    if not heading_path:
        return text
    context = " > ".join(heading_path)
    return text if text.startswith("#") else f"{context}\n\n{text}"
