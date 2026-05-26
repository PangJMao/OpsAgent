from __future__ import annotations

import re

from ops_agent.config.retrieval_config import DEFAULT_RETRIEVAL_CONFIG, RetrievalConfig
from ops_agent.services.retriever.schema import HybridCandidate


class MmrSelector:
    """MMR 多样性选择：兼顾相关性与来源、主题、场景差异。"""

    def __init__(self, config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG) -> None:
        self.config = config

    def select(self, candidates: list[HybridCandidate], top_k: int) -> list[HybridCandidate]:
        selected: list[HybridCandidate] = []
        remaining = list(candidates)
        while remaining and len(selected) < top_k:
            best = max(remaining, key=lambda item: self._mmr_score(item, selected))
            selected.append(best)
            remaining.remove(best)
        return selected

    def _mmr_score(self, candidate: HybridCandidate, selected: list[HybridCandidate]) -> float:
        relevance = candidate.rerank_score or candidate.hybrid_score
        diversity_penalty = max((_similarity(candidate, item) for item in selected), default=0.0)
        return (self.config.mmr_lambda * relevance) - ((1 - self.config.mmr_lambda) * diversity_penalty)


def _similarity(left: HybridCandidate, right: HybridCandidate) -> float:
    score = _jaccard(_tokens(left.hit.chunk.text), _tokens(right.hit.chunk.text))
    for field in ("doc_name", "sheet_name", "topic", "business_scene"):
        if left.hit.chunk.metadata.get(field) and left.hit.chunk.metadata.get(field) == right.hit.chunk.metadata.get(field):
            score += 0.15
    return min(score, 1.0)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", text))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
