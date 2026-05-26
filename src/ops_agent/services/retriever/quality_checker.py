from __future__ import annotations

from dataclasses import dataclass, field

from ops_agent.config.retrieval_config import DEFAULT_RETRIEVAL_CONFIG, RetrievalConfig
from ops_agent.services.document_processing.cleaners import mojibake_ratio
from ops_agent.services.retriever.schema import HybridCandidate


@dataclass
class QualityDecision:
    can_answer: bool
    confidence: str
    reason: str
    flags: list[str] = field(default_factory=list)


class QualityChecker:
    def __init__(self, config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG) -> None:
        self.config = config

    def check(self, candidates: list[HybridCandidate]) -> QualityDecision:
        if not candidates:
            return QualityDecision(False, "low", "知识库未检索到相关内容", ["no_results"])
        clean_candidates = [candidate for candidate in candidates if mojibake_ratio(candidate.hit.chunk.text) <= 0.08]
        if not clean_candidates:
            return QualityDecision(False, "low", "检索结果疑似乱码或解析异常，需要重新入库", ["mojibake"])
        top_score = max(candidate.rerank_score or candidate.hybrid_score for candidate in clean_candidates)
        flags = []
        if any(str(candidate.hit.chunk.metadata.get("risk_level")) == "high" for candidate in clean_candidates):
            flags.append("high_risk_content")
        if top_score < self.config.medium_confidence_threshold:
            return QualityDecision(False, "low", "知识库没有找到明确依据", [*flags, "low_confidence"])
        if top_score < self.config.high_confidence_threshold:
            return QualityDecision(True, "medium", "根据当前知识库可归纳回答", [*flags, "medium_confidence"])
        return QualityDecision(True, "high", "可基于知识库回答", flags)
