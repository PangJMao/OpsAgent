from __future__ import annotations

import re
from typing import Protocol

from ops_agent.models import RetrievalHit
from ops_agent.services.retriever.schema import HybridCandidate

TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}")
BOOST_FIELDS = ("topic", "section", "doc_name", "doc_type", "sheet_name", "business_scene", "applicable_stage")


class KeywordSearchStore(Protocol):
    def keyword_search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        ...


class KeywordRetriever:
    """关键词/全文检索包装层，兼容 PostgreSQL FTS、ILike 或本地 BM25 实现。"""

    def __init__(self, store: KeywordSearchStore) -> None:
        self.store = store

    def retrieve(self, queries: list[str], top_k: int) -> list[HybridCandidate]:
        if not hasattr(self.store, "keyword_search"):
            return []
        candidates: list[HybridCandidate] = []
        for query in queries:
            keywords = extract_keywords(query)
            for hit in self.store.keyword_search(query, top_k=top_k):
                matched = _matched_keywords(hit, keywords)
                score = max(float(hit.score), 0.0) + _field_boost(hit, matched)
                candidates.append(
                    HybridCandidate(
                        hit=hit,
                        keyword_score=score,
                        matched_keywords=matched,
                        sources=["keyword"],
                    )
                )
        return candidates


def extract_keywords(query: str) -> list[str]:
    tokens = TOKEN_RE.findall(query)
    expanded = []
    for token in tokens:
        expanded.append(token)
        if token == "沟通":
            expanded.extend(["安抚", "认可鼓励", "语气", "语速", "承诺", "合规"])
        if token == "客户":
            expanded.extend(["投诉", "敏感词", "联系人"])
    return _unique([token for token in expanded if len(token) >= 2])


def _matched_keywords(hit: RetrievalHit, keywords: list[str]) -> list[str]:
    metadata_text = " ".join(str(hit.chunk.metadata.get(field) or "") for field in BOOST_FIELDS)
    haystack = f"{hit.chunk.text} {metadata_text}"
    return [keyword for keyword in keywords if keyword in haystack]


def _field_boost(hit: RetrievalHit, matched: list[str]) -> float:
    if not matched:
        return 0.0
    boost = 0.0
    for field in BOOST_FIELDS:
        value = str(hit.chunk.metadata.get(field) or "")
        if any(keyword in value for keyword in matched):
            boost += 1.0
    return boost


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
