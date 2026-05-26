from __future__ import annotations

from typing import Protocol

from ops_agent.models import RetrievalHit
from ops_agent.services.retriever.schema import HybridCandidate


class VectorSearchStore(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        ...


class VectorRetriever:
    """对每个 rewrite query 执行向量检索，并把分数保存在内部候选对象里。"""

    def __init__(self, store: VectorSearchStore) -> None:
        self.store = store

    def retrieve(self, queries: list[str], top_k: int) -> list[HybridCandidate]:
        candidates: list[HybridCandidate] = []
        for query in queries:
            for hit in self.store.search(query, top_k=top_k):
                candidates.append(HybridCandidate(hit=hit, vector_score=max(float(hit.score), 0.0), sources=["vector"]))
        return candidates
