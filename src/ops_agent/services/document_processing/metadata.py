from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
import re

from ops_agent.services.document_processing.schema import DocumentChunk, IngestedChunk

INGESTION_VERSION = "document-processing-v1"


class MetadataBuilder:
    def build(self, chunks: list[DocumentChunk]) -> list[IngestedChunk]:
        return [self.build_one(chunk) for chunk in chunks]

    def build_one(self, chunk: DocumentChunk) -> IngestedChunk:
        source_key = str(Path(chunk.source).resolve())
        document_id = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]
        content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        location_key = "|".join(
            [
                source_key,
                str(chunk.page or ""),
                str(chunk.sheet_name or ""),
                str(chunk.row_index or ""),
                str(chunk.slide_number or ""),
                str(chunk.chunk_index),
            ]
        )
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, location_key))
        metadata = {
            "chunk_id": chunk_id,
            "source": chunk.source,
            "doc_name": chunk.file_name,
            "doc_type": _infer_doc_type(chunk.file_name, chunk.text, chunk.metadata),
            "source_type": chunk.file_type,
            "file_name": chunk.file_name,
            "file_type": chunk.file_type,
            "page": chunk.page,
            "sheet_name": chunk.sheet_name,
            "row_index": chunk.row_index,
            "source_row": chunk.row_index or chunk.page or chunk.slide_number,
            "slide_number": chunk.slide_number,
            "section": chunk.section,
            "business_scene": str(chunk.metadata.get("business_scene") or _infer_business_scene(chunk.text)),
            "topic": str(chunk.metadata.get("topic") or _infer_topic(chunk.text)),
            "risk_level": str(chunk.metadata.get("risk_level") or _infer_risk_level(chunk.text)),
            "applicable_stage": str(chunk.metadata.get("applicable_stage") or _infer_stage(chunk.text)),
            "heading_path": chunk.heading_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "content_hash": content_hash,
            "parser": chunk.parser,
            "version": INGESTION_VERSION,
            "chunk_strategy": chunk.strategy,
            "chunk_index": chunk.chunk_index,
            **chunk.metadata,
        }
        return IngestedChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            title=chunk.file_name,
            text=chunk.text,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            metadata=metadata,
        )


def _infer_doc_type(file_name: str, text: str, metadata: dict[str, object]) -> str:
    value = str(metadata.get("doc_type") or "")
    if value:
        return value
    joined = f"{file_name} {text}"
    if any(word in joined for word in ("法务", "诉讼", "律师函", "诉前")):
        return "法务话术"
    if any(word in joined for word in ("沟通", "安抚", "鼓励", "联系人", "话术")):
        return "沟通话术"
    if any(word in joined for word in ("流程", "操作", "步骤")):
        return "操作流程"
    if any(word in joined for word in ("制度", "规范", "政策")):
        return "制度规范"
    return "知识文档"


def _infer_business_scene(text: str) -> str:
    for marker in ("客户不满", "投诉", "敏感词", "联系人", "法务沟通", "诉前沟通", "还款意愿", "核资"):
        if marker in text:
            return marker
    return ""


def _infer_topic(text: str) -> str:
    for marker in ("安抚", "认可鼓励", "敏感词回应", "联系人沟通", "身份核实", "合规要求", "阶段话术"):
        if marker in text:
            return marker
    return ""


def _infer_risk_level(text: str) -> str:
    if any(marker in text for marker in ("法务", "诉讼", "律师函", "威胁", "征信", "报警", "敏感词")):
        return "high"
    if any(marker in text for marker in ("投诉", "联系人", "承诺", "不满", "合规")):
        return "medium"
    return "low"


def _infer_stage(text: str) -> str:
    match = re.search(r"D\d+\s*[-~至到]\s*D?\d+", text, flags=re.IGNORECASE)
    return match.group(0) if match else ""
