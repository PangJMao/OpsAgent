from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    source_path: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedDocument:
    title: str
    markdown: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Citation:
    document_id: str
    title: str
    chunk_id: str
    score: float
    heading_path: list[str] = field(default_factory=list)
    chunk_strategy: str = ""


@dataclass(frozen=True)
class RagAnswer:
    trace_id: str
    question: str
    answer: str
    citations: list[Citation]
    confidence: float
    refused: bool


RouteType = Literal["knowledge_qa", "tool_call", "hybrid"]
RiskLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class AgentReview:
    passed: bool
    risk_level: RiskLevel
    issues: list[str] = field(default_factory=list)
    final_answer: str = ""


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    tool: str
    ok: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class AgentAnswer:
    trace_id: str
    question: str
    route: RouteType
    answer: str
    citations: list[Citation]
    confidence: float
    refused: bool
    tool_results: list[ToolResult] = field(default_factory=list)
    review: AgentReview | None = None
