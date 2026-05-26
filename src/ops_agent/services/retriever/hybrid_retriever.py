from __future__ import annotations

from dataclasses import asdict

from ops_agent.config.retrieval_config import DEFAULT_RETRIEVAL_CONFIG, RetrievalConfig
from ops_agent.models import Chunk, RetrievalHit
from ops_agent.services.rerank_service import Reranker
from ops_agent.services.retriever.context_compressor import ContextCompressor
from ops_agent.services.retriever.deduplicator import Deduplicator
from ops_agent.services.retriever.keyword_retriever import KeywordRetriever
from ops_agent.services.retriever.metadata_filter import MetadataFilter
from ops_agent.services.retriever.mmr import MmrSelector
from ops_agent.services.retriever.quality_checker import QualityChecker
from ops_agent.services.retriever.query_rewriter import QueryRewriter, has_exact_business_terms
from ops_agent.services.retriever.schema import HybridCandidate, RetrievalResult
from ops_agent.services.retriever.vector_retriever import VectorRetriever


class HybridRetriever:
    """企业 RAG 混合检索主类：vector + keyword + metadata + rerank + dedupe + MMR。"""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_retriever: KeywordRetriever,
        reranker: Reranker,
        config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.reranker = reranker
        self.config = config
        self.rewriter = QueryRewriter(config)
        self.metadata_filter = MetadataFilter()
        self.deduplicator = Deduplicator(config)
        self.mmr = MmrSelector(config)
        self.quality_checker = QualityChecker(config)
        self.compressor = ContextCompressor()

    def retrieve(self, question: str, question_type: str = "knowledge_qa", filters: dict | None = None) -> RetrievalResult:
        queries = self.rewriter.rewrite(question, question_type)
        vector_candidates = self.vector_retriever.retrieve(queries, top_k=self.config.per_query_top_k)
        keyword_candidates = self.keyword_retriever.retrieve(queries, top_k=self.config.per_query_top_k)
        merged = merge_results([*vector_candidates, *keyword_candidates])
        filtered = self.metadata_filter.apply(merged, question, question_type)
        normalize_scores(filtered)
        ranked = hybrid_rank(filtered, question, self.config)[: self.config.hybrid_pool_size]
        reranked = rerank_candidates(ranked, question, self.reranker, self.config.rerank_top_k)
        deduped = self.deduplicator.deduplicate(reranked)
        selected = self.mmr.select(deduped, top_k=self.config.final_top_k)
        quality = self.quality_checker.check(selected)
        contexts = self.compressor.compress(selected)
        return RetrievalResult(
            can_answer=quality.can_answer,
            confidence=quality.confidence,
            reason=quality.reason,
            contexts=contexts,
            hits=[candidate.hit for candidate in selected],
            debug={
                "rewritten_queries": queries,
                "raw_vector_results_count": len(vector_candidates),
                "raw_keyword_results_count": len(keyword_candidates),
                "after_merge_count": len(merged),
                "after_rerank_count": len(reranked),
                "after_mmr_count": len(selected),
                "quality": asdict(quality),
            },
        )


def merge_results(candidates: list[HybridCandidate]) -> list[HybridCandidate]:
    merged: dict[str, HybridCandidate] = {}
    for candidate in candidates:
        chunk_id = candidate.hit.chunk.chunk_id
        existing = merged.get(chunk_id)
        if existing is None:
            merged[chunk_id] = candidate
            continue
        existing.vector_score = max(existing.vector_score, candidate.vector_score)
        existing.keyword_score = max(existing.keyword_score, candidate.keyword_score)
        existing.matched_keywords = sorted(set(existing.matched_keywords + candidate.matched_keywords))
        existing.sources = sorted(set(existing.sources + candidate.sources))
        existing.hit = existing.hit if existing.hit.score >= candidate.hit.score else candidate.hit
    return list(merged.values())


def normalize_scores(candidates: list[HybridCandidate]) -> None:
    _normalize(candidates, "vector_score")
    _normalize(candidates, "keyword_score")
    _normalize(candidates, "metadata_score")
    _normalize(candidates, "rerank_score")


def hybrid_rank(candidates: list[HybridCandidate], question: str, config: RetrievalConfig) -> list[HybridCandidate]:
    exact = has_exact_business_terms(question)
    has_vector = any("vector" in candidate.sources for candidate in candidates)
    has_keyword = any("keyword" in candidate.sources for candidate in candidates)
    if has_vector and not has_keyword:
        vector_weight, keyword_weight, metadata_weight, source_weight = 0.80, 0.0, 0.15, 0.05
    elif has_keyword and not has_vector:
        vector_weight, keyword_weight, metadata_weight, source_weight = 0.0, 0.80, 0.15, 0.05
    else:
        vector_weight = config.exact_vector_weight if exact else config.vector_weight
        keyword_weight = config.exact_keyword_weight if exact else config.keyword_weight
        metadata_weight = config.metadata_weight
        source_weight = config.source_bonus_weight
    for candidate in candidates:
        source_bonus = 1.0 if {"vector", "keyword"}.issubset(set(candidate.sources)) else 0.5
        candidate.hybrid_score = (
            vector_weight * candidate.vector_score
            + keyword_weight * candidate.keyword_score
            + metadata_weight * candidate.metadata_score
            + source_weight * source_bonus
        )
        candidate.hit = _with_score(candidate.hit, candidate.hybrid_score)
    return sorted(candidates, key=lambda item: item.hybrid_score, reverse=True)


def rerank_candidates(candidates: list[HybridCandidate], question: str, reranker: Reranker, top_k: int) -> list[HybridCandidate]:
    if not candidates:
        return []
    hits = [_rerank_hit(candidate) for candidate in candidates]
    reranked_hits = reranker.rerank(question, hits, top_k=top_k)
    by_id = {candidate.hit.chunk.chunk_id: candidate for candidate in candidates}
    reranked: list[HybridCandidate] = []
    for index, hit in enumerate(reranked_hits):
        candidate = by_id.get(hit.chunk.chunk_id)
        if candidate is None:
            continue
        candidate.rerank_score = max(float(hit.score), 0.0)
        candidate.hit = _with_score(candidate.hit, candidate.rerank_score or candidate.hybrid_score)
        reranked.append(candidate)
    normalize_scores(reranked)
    return sorted(reranked, key=lambda item: item.rerank_score or item.hybrid_score, reverse=True)


def _rerank_hit(candidate: HybridCandidate) -> RetrievalHit:
    metadata = candidate.hit.chunk.metadata
    parts = [
        str(metadata.get("doc_name") or candidate.hit.chunk.title),
        str(metadata.get("topic") or ""),
        str(metadata.get("section") or ""),
        candidate.hit.chunk.text[:900],
    ]
    chunk = Chunk(
        chunk_id=candidate.hit.chunk.chunk_id,
        document_id=candidate.hit.chunk.document_id,
        title=candidate.hit.chunk.title,
        text="\n".join(part for part in parts if part),
        start_char=candidate.hit.chunk.start_char,
        end_char=candidate.hit.chunk.end_char,
        metadata=metadata,
    )
    return RetrievalHit(chunk=chunk, score=candidate.hybrid_score)


def _with_score(hit: RetrievalHit, score: float) -> RetrievalHit:
    return RetrievalHit(chunk=hit.chunk, score=score)


def _normalize(candidates: list[HybridCandidate], attr: str) -> None:
    values = [float(getattr(candidate, attr)) for candidate in candidates]
    if not values:
        return
    lo = min(values)
    hi = max(values)
    for candidate in candidates:
        value = float(getattr(candidate, attr))
        setattr(candidate, attr, min(max(value, 0.0), 1.0) if hi == lo else (value - lo) / (hi - lo))
