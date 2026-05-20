from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NormalizedDocument:
    title: str
    markdown: str
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_to_markdown(path: Path) -> NormalizedDocument:
    """把不同来源的文档归一化为 Markdown。

    当前阶段先支持 Markdown 和纯文本。后续 PDF、DOCX、HTML 等格式只需要在这里扩展，
    下游切分器仍然只处理统一的 Markdown。
    """

    suffix = path.suffix.lower()
    if suffix == ".md":
        return _normalize_markdown(path)
    if suffix == ".txt":
        return _normalize_text(path)
    raise ValueError(f"暂不支持归一化该文档类型：{suffix}")


def _normalize_markdown(path: Path) -> NormalizedDocument:
    markdown = path.read_text(encoding="utf-8").strip()
    title = _first_heading(markdown) or path.stem
    return NormalizedDocument(
        title=title,
        markdown=markdown,
        metadata={"source_format": "markdown", "normalizer": "markdown_passthrough"},
    )


def _normalize_text(path: Path) -> NormalizedDocument:
    text = path.read_text(encoding="utf-8").strip()
    title = path.stem
    markdown = f"# {title}\n\n{text}" if text else f"# {title}"
    return NormalizedDocument(
        title=title,
        markdown=markdown,
        metadata={"source_format": "text", "normalizer": "text_to_markdown"},
    )


def _first_heading(markdown: str) -> str | None:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or None
    return None
