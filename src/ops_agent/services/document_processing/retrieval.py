from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ops_agent.config import settings
from ops_agent.models import Chunk, RetrievalHit
from ops_agent.services.rerank_service import Reranker, create_reranker
from ops_agent.services.vector_store import LocalVectorStore, PgVectorStore, create_vector_store


class SearchStore(Protocol):
    def search(self, query: str, top_k: int = settings.top_k) -> list[RetrievalHit]:
        ...

    def keyword_search(self, query: str, top_k: int = settings.top_k) -> list[RetrievalHit]:
        ...


@dataclass(frozen=True)
class HybridSearchConfig:
    vector_weight: float = settings.hybrid_vector_weight
    bm25_weight: float = settings.hybrid_bm25_weight
    recall_multiplier: int = 4


class BM25KeywordStore:
    def __init__(self, store: SearchStore | None = None) -> None:
        self.store = store or create_vector_store()

    def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        return self.store.keyword_search(query, top_k=top_k)


class HybridRetriever:
    def __init__(
        self,
        store: SearchStore | None = None,
        reranker: Reranker | None = None,
        config: HybridSearchConfig | None = None,
    ) -> None:
        self.store = store or create_vector_store()
        self.reranker = reranker or create_reranker()
        self.config = config or HybridSearchConfig()

    def search(self, query: str, top_k: int = 10, rerank: bool = True) -> list[RetrievalHit]:
        recall_k = max(top_k * self.config.recall_multiplier, top_k)
        vector_hits = self.store.search(query, top_k=recall_k)
        keyword_hits = self.store.keyword_search(query, top_k=recall_k)
        merged = self.merge(vector_hits=vector_hits, keyword_hits=keyword_hits, top_k=recall_k)
        if rerank:
            return self.reranker.rerank(query, merged, top_k=top_k)
        return merged[:top_k]

    def merge(
        self,
        vector_hits: list[RetrievalHit],
        keyword_hits: list[RetrievalHit],
        top_k: int,
    ) -> list[RetrievalHit]:
        vector_scores = _normalize_by_chunk_id(vector_hits)
        keyword_scores = _normalize_by_chunk_id(keyword_hits)
        chunks: dict[str, Chunk] = {}
        for hit in [*vector_hits, *keyword_hits]:
            chunks[hit.chunk.chunk_id] = hit.chunk

        merged: list[RetrievalHit] = []
        for chunk_id, chunk in chunks.items():
            vector_score = vector_scores.get(chunk_id, 0.0)
            bm25_score = keyword_scores.get(chunk_id, 0.0)
            score = (vector_score * self.config.vector_weight) + (bm25_score * self.config.bm25_weight)
            metadata = {
                **chunk.metadata,
                "vector_score": round(vector_score, 6),
                "bm25_score": round(bm25_score, 6),
                "hybrid_score": round(score, 6),
                "retrieval_mode": "hybrid",
            }
            merged.append(
                RetrievalHit(
                    chunk=Chunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        title=chunk.title,
                        text=chunk.text,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        metadata=metadata,
                    ),
                    score=score,
                )
            )
        merged.sort(key=lambda hit: hit.score, reverse=True)
        return merged[:top_k]


class RetrievalReranker:
    def __init__(self, reranker: Reranker | None = None) -> None:
        self.reranker = reranker or create_reranker()

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int = 10) -> list[RetrievalHit]:
        return self.reranker.rerank(query, hits, top_k=top_k)


def create_retriever(store: LocalVectorStore | PgVectorStore | None = None) -> HybridRetriever | LocalVectorStore | PgVectorStore:
    vector_store = store or create_vector_store()
    if settings.retrieval_mode.lower() == "hybrid":
        return HybridRetriever(store=vector_store)
    return vector_store


def _normalize_by_chunk_id(hits: list[RetrievalHit]) -> dict[str, float]:
    if not hits:
        return {}
    max_score = max(hit.score for hit in hits)
    min_score = min(hit.score for hit in hits)
    if max_score == min_score:
        return {hit.chunk.chunk_id: 1.0 for hit in hits if hit.score > 0}
    return {hit.chunk.chunk_id: (hit.score - min_score) / (max_score - min_score) for hit in hits}
