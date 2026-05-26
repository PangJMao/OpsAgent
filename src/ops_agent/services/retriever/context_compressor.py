from __future__ import annotations

import re

from ops_agent.services.document_processing.cleaners import clean_text
from ops_agent.services.retriever.schema import CompressedContext, HybridCandidate


class ContextCompressor:
    """Compress final candidates into clean business-facing context."""

    def compress(self, candidates: list[HybridCandidate]) -> list[CompressedContext]:
        contexts: list[CompressedContext] = []
        seen: set[tuple[str, str, tuple[str, ...]]] = set()
        for candidate in candidates:
            metadata = candidate.hit.chunk.metadata
            key_points = _key_points(candidate.hit.chunk.text)
            key = (_source_label(candidate), str(metadata.get("topic") or ""), tuple(key_points))
            if key in seen:
                continue
            seen.add(key)
            contexts.append(
                CompressedContext(
                    source=_source_label(candidate),
                    topic=str(metadata.get("topic") or metadata.get("business_scene") or metadata.get("section") or ""),
                    key_points=key_points,
                    risk_level=str(metadata.get("risk_level") or "low"),
                )
            )
        return contexts


def _source_label(candidate: HybridCandidate) -> str:
    metadata = candidate.hit.chunk.metadata
    doc_name = str(metadata.get("doc_name") or metadata.get("file_name") or candidate.hit.chunk.title)
    doc_name = re.sub(r"\.[A-Za-z0-9]+$", "", doc_name)
    parts = [f"《{doc_name}》"]
    if metadata.get("sheet_name"):
        parts.append(f"Sheet：{metadata['sheet_name']}")
    section = metadata.get("section")
    heading_path = metadata.get("heading_path") or []
    if not section and isinstance(heading_path, list) and heading_path:
        section = " > ".join(str(item) for item in heading_path)
    if section:
        parts.append(f"章节：{section}")
    if metadata.get("page"):
        parts.append(f"第 {metadata['page']} 页")
    if metadata.get("row_index") or metadata.get("source_row"):
        parts.append(f"第 {metadata.get('row_index') or metadata.get('source_row')} 行")
    return " ".join(parts)


def _key_points(text: str) -> list[str]:
    cleaned = clean_text(text)
    pieces = re.split(r"[\n。；;.!?？]+", cleaned)
    points = []
    for piece in pieces:
        point = _clean_point(piece)
        if not point or point in points:
            continue
        points.append(point[:220])
        if len(points) >= 4:
            break
    return points or [_clean_point(cleaned)[:220]]


def _clean_point(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip(" -\t|")
    if not text:
        return ""
    if _looks_mojibake(text) or _looks_like_table_noise(text) or _looks_like_generic_call_flow(text):
        return ""
    noise_prefixes = (
        "业务分类",
        "字段",
        "column_",
        "Sheet",
        "序号",
        "编号",
        "DO-",
        "do-",
        "参考话术",
        "鏁欐嵁",
    )
    if any(text.startswith(prefix) for prefix in noise_prefixes):
        return ""
    if text.count("(") > text.count(")") or text.count("（") > text.count("）"):
        text = text.replace("(", "").replace("（", "")
    text = re.sub(r"^(?:\d+[、.])+", "", text).strip()
    return text if len(text) >= 4 else ""


def _looks_like_table_noise(text: str) -> bool:
    lowered = text.lower()
    if "column_" in lowered:
        return True
    if "|" in text and (text.count("|") >= 2 or re.search(r"\|\s*\|", text)):
        return True
    if text.count("为本人/联系人") >= 2 or text.count("本人/联系人") >= 3:
        return True
    if "零容忍" in text and len(text) < 30:
        return True
    if "涉及任意一项" in text and "0" in text and "分" in text:
        return True
    if text.count(",") >= 4 and ("column" in lowered or "本人" in text):
        return True
    return False


def _looks_mojibake(text: str) -> bool:
    markers = (
        "Root Entry",
        "SummaryInformation",
        "DocumentSummaryInformation",
        "WordDocument",
        "\ufffd",
        "閿?",
        "锟?",
    )
    if any(marker in text for marker in markers):
        return True
    suspicious = sum(1 for char in text if ord(char) < 32 and char not in "\t\n\r")
    return suspicious / max(len(text), 1) > 0.03


def _looks_like_generic_call_flow(text: str) -> bool:
    markers = (
        "委外",
        "受宜享花",
        "结束语",
        "拜拜",
        "核身",
        "核实姓名",
        "参考话术",
        "张三",
        "李四",
        "先生/女士",
        "家人朋友",
        "请问您是",
        "请问是",
        "核对三方号码",
        "何时办理",
        "何地办理",
        "使用期间是否停机",
        "用户画像",
        "年龄\\户籍",
        "当天非首通电话",
        "自报家门",
        "主动挂机",
        "Don't-0分",
    )
    return any(marker in text for marker in markers)
