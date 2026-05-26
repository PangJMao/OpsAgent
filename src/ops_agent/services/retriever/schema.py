from __future__ import annotations

from dataclasses import dataclass, field

from ops_agent.models import RetrievalHit


@dataclass
class HybridCandidate:
    hit: RetrievalHit
    vector_score: float = 0.0
    keyword_score: float = 0.0
    metadata_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    matched_keywords: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CompressedContext:
    source: str
    topic: str
    key_points: list[str]
    risk_level: str


@dataclass
class RetrievalResult:
    can_answer: bool
    confidence: str
    reason: str
    contexts: list[CompressedContext]
    hits: list[RetrievalHit]
    debug: dict[str, object] = field(default_factory=dict)
