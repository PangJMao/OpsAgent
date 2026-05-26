from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParsedBlock:
    text: str
    block_type: str = "text"
    page: int | None = None
    sheet_name: str | None = None
    row_index: int | None = None
    slide_number: int | None = None
    section: str | None = None
    heading_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    source: str
    file_name: str
    file_type: str
    title: str
    parser: str
    blocks: list[ParsedBlock]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return Path(self.source)


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    source: str
    file_name: str
    file_type: str
    page: int | None = None
    sheet_name: str | None = None
    row_index: int | None = None
    slide_number: int | None = None
    section: str | None = None
    heading_path: list[str] = field(default_factory=list)
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0
    strategy: str = ""
    parser: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestedChunk:
    chunk_id: str
    document_id: str
    title: str
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any]
