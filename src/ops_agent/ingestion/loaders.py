from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from ops_agent.core.config import settings
from ops_agent.ingestion.normalizers import normalize_to_markdown
from ops_agent.schemas import Document

SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_text_document(path: Path) -> Document:
    """读取原始文档，并先归一化成 Markdown 中间表示。"""

    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"不支持的文档类型：{path.suffix!r}。当前支持：{supported}")

    normalized = normalize_to_markdown(path)
    document_id = hashlib.sha256(
        f"{path.name}:{normalized.markdown}".encode("utf-8")
    ).hexdigest()[:16]
    return Document(
        document_id=document_id,
        title=normalized.title,
        source_path=str(path),
        content=normalized.markdown,
        metadata={
            **normalized.metadata,
            "source_suffix": path.suffix.lower(),
            "normalized_format": "markdown",
            "bytes": path.stat().st_size,
        },
    )


def persist_source_document(path: Path, document_id: str) -> Path:
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    target = settings.documents_dir / f"{document_id}{path.suffix.lower()}"
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)
    return target


def persist_normalized_markdown(document: Document) -> Path:
    """保存归一化后的 Markdown，方便排查转换和切分质量。"""

    settings.normalized_dir.mkdir(parents=True, exist_ok=True)
    target = settings.normalized_dir / f"{document.document_id}.md"
    target.write_text(document.content, encoding="utf-8")
    return target
