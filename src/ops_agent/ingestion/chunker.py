from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ops_agent.core.config import settings
from ops_agent.schemas import Chunk, Document

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class MarkdownSection:
    text: str
    start_char: int
    end_char: int
    heading_path: list[str]
    heading_level: int


def chunk_document(document: Document) -> list[Chunk]:
    """优先按 Markdown 标题切分；标题缺失或章节过长时用重叠窗口兜底。"""

    markdown = document.content.strip()
    if not markdown:
        return []

    sections = _split_markdown_sections(markdown)
    if not sections:
        return _window_chunks(
            document=document,
            text=markdown,
            base_start=0,
            heading_path=[],
            heading_level=0,
            strategy="overlap_window_fallback",
            fallback_used=True,
        )

    chunks: list[Chunk] = []
    for section in sections:
        if len(section.text) <= settings.chunk_size:
            chunks.append(
                _build_chunk(
                    document=document,
                    text=_with_heading_context(section.text, section.heading_path),
                    start_char=section.start_char,
                    end_char=section.end_char,
                    heading_path=section.heading_path,
                    heading_level=section.heading_level,
                    strategy="markdown_heading",
                    fallback_used=False,
                    chunk_index=len(chunks),
                )
            )
            continue

        chunks.extend(
            _window_chunks(
                document=document,
                text=section.text,
                base_start=section.start_char,
                heading_path=section.heading_path,
                heading_level=section.heading_level,
                strategy="markdown_heading_window_fallback",
                fallback_used=True,
                start_index=len(chunks),
            )
        )

    return chunks


def _split_markdown_sections(markdown: str) -> list[MarkdownSection]:
    matches = list(HEADING_RE.finditer(markdown))
    if not matches:
        return []

    sections: list[MarkdownSection] = []
    heading_stack: list[tuple[int, str]] = []

    for index, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)

        heading_stack = [
            (item_level, item_title)
            for item_level, item_title in heading_stack
            if item_level < level
        ]
        heading_stack.append((level, title))
        heading_path = [item_title for _, item_title in heading_stack]

        section_text = markdown[start:end].strip()
        if section_text:
            sections.append(
                MarkdownSection(
                    text=section_text,
                    start_char=start,
                    end_char=end,
                    heading_path=heading_path,
                    heading_level=level,
                )
            )

    return sections


def _window_chunks(
    document: Document,
    text: str,
    base_start: int,
    heading_path: list[str],
    heading_level: int,
    strategy: str,
    fallback_used: bool,
    start_index: int = 0,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0

    while start < len(text):
        end = min(start + settings.chunk_size, len(text))
        candidate = text[start:end]

        # 兜底窗口仍然优先在自然边界结束，避免把一句话或一个段落切断。
        boundary = max(candidate.rfind("\n\n"), candidate.rfind("。"), candidate.rfind("."))
        if boundary > settings.chunk_size * 0.55 and end < len(text):
            end = start + boundary + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                _build_chunk(
                    document=document,
                    text=_with_heading_context(chunk_text, heading_path),
                    start_char=base_start + start,
                    end_char=base_start + end,
                    heading_path=heading_path,
                    heading_level=heading_level,
                    strategy=strategy,
                    fallback_used=fallback_used,
                    chunk_index=start_index + len(chunks),
                )
            )

        if end >= len(text):
            break
        start = max(0, end - settings.chunk_overlap)

    return chunks


def _with_heading_context(text: str, heading_path: list[str]) -> str:
    if not heading_path:
        return text
    return f"{' > '.join(heading_path)}\n\n{text}"


def _build_chunk(
    document: Document,
    text: str,
    start_char: int,
    end_char: int,
    heading_path: list[str],
    heading_level: int,
    strategy: str,
    fallback_used: bool,
    chunk_index: int,
) -> Chunk:
    chunk_hash = hashlib.sha256(
        f"{document.document_id}:{chunk_index}:{text}".encode("utf-8")
    ).hexdigest()[:16]
    return Chunk(
        chunk_id=f"{document.document_id}-{chunk_hash}",
        document_id=document.document_id,
        title=document.title,
        text=text,
        start_char=start_char,
        end_char=end_char,
        metadata={
            **document.metadata,
            "chunk_index": chunk_index,
            "heading_path": heading_path,
            "heading_level": heading_level,
            "chunk_strategy": strategy,
            "fallback_used": fallback_used,
        },
    )
