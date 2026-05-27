from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


INTERNAL_PATTERNS = (
    "chunk_id",
    "vector_score",
    "rerank_score",
    "hybrid_score",
    "embedding",
    "数据库 UUID",
    "Sheet3 Sheet3",
    "业务分类",
    "DO-1分",
    "字段3",
    "Root Entry",
    "SummaryInformation",
    "DocumentSummaryInformation",
    "WordDocument",
)

UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
SCORE_RE = re.compile(r"\b(?:score|相似度|相关度)\s*[:：=]\s*\d+(?:\.\d+)?\b", re.IGNORECASE)


@dataclass
class EvalResult:
    case_id: str
    question: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    intent: str | None = None
    risk_level: str | None = None
    refused: bool = False


def evaluate_case(case: dict[str, Any], response: dict[str, Any]) -> EvalResult:
    answer = _extract_answer(response)
    sources = _extract_sources(response, answer)
    intent = _extract_first(response, "intent", "question_type", "expected_intent")
    risk_level = _extract_first(response, "risk_level", "riskLevel")
    refused = bool(response.get("refused", False))
    failures: list[str] = []

    if not answer.strip():
        failures.append("回答为空")

    if refused and not case.get("allow_refused", False):
        failures.append("错误拒答")

    for keyword in case.get("must_contain") or []:
        if keyword not in answer:
            failures.append(f"缺少关键词：{keyword}")

    for keyword in case.get("must_not_contain") or []:
        if keyword in answer:
            failures.append(f"不应出现：{keyword}")

    source_text = "\n".join(sources)
    for keyword in case.get("expected_sources") or []:
        if keyword not in source_text and keyword not in answer:
            failures.append(f"缺少引用来源：{keyword}")

    for keyword in case.get("forbidden_sources") or []:
        if keyword in source_text:
            failures.append(f"不应引用来源：{keyword}")

    expected_intent = case.get("expected_intent")
    if expected_intent and intent and expected_intent != intent:
        failures.append(f"意图不匹配：期望 {expected_intent}，实际 {intent}")

    expected_risk_level = case.get("expected_risk_level")
    if expected_risk_level and risk_level and expected_risk_level != risk_level:
        failures.append(f"风险等级不匹配：期望 {expected_risk_level}，实际 {risk_level}")

    failures.extend(_internal_field_failures(answer))

    return EvalResult(
        case_id=str(case.get("id") or ""),
        question=str(case.get("question") or ""),
        passed=not failures,
        failures=failures,
        answer=answer,
        sources=sources,
        intent=intent,
        risk_level=risk_level,
        refused=refused,
    )


def _extract_answer(response: dict[str, Any]) -> str:
    for key in ("answer", "content", "message", "text"):
        value = response.get(key)
        if isinstance(value, str):
            return value
    data = response.get("data")
    if isinstance(data, dict):
        return _extract_answer(data)
    return ""


def _extract_sources(response: dict[str, Any], answer: str) -> list[str]:
    sources: list[str] = []
    citations = response.get("citations") or response.get("sources") or []
    if isinstance(citations, list):
        for item in citations:
            if isinstance(item, str):
                sources.append(item)
            elif isinstance(item, dict):
                parts = [str(item.get(key) or "") for key in ("title", "doc_name", "source", "section", "sheet_name")]
                source = " ".join(part for part in parts if part)
                if source:
                    sources.append(source)
    if "引用来源" in answer:
        sources.append(answer.split("引用来源", 1)[1])
    return sources


def _extract_first(response: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = response.get(key)
        if isinstance(value, str) and value:
            return value
    data = response.get("data")
    if isinstance(data, dict):
        return _extract_first(data, *keys)
    return None


def _internal_field_failures(answer: str) -> list[str]:
    failures: list[str] = []
    lowered = answer.lower()
    for marker in INTERNAL_PATTERNS:
        if marker.lower() in lowered:
            failures.append(f"出现内部字段：{marker}")
    if UUID_RE.search(answer):
        failures.append("出现内部字段：UUID")
    if SCORE_RE.search(answer):
        failures.append("出现内部字段：score/相关度")
    return failures
