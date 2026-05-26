from __future__ import annotations

import importlib
import re
from collections.abc import Iterable
from typing import Protocol

from ops_agent.config import settings
from ops_agent.models import RetrievalHit

TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int = settings.rerank_top_k) -> list[RetrievalHit]:
        ...


class BgeReranker:
    """BGE reranker backed by FlagEmbedding.FlagReranker."""

    def __init__(
        self,
        model_name: str = settings.rerank_model,
        use_fp16: bool = settings.rerank_use_fp16,
    ) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model: object | None = None

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int = settings.rerank_top_k) -> list[RetrievalHit]:
        if not hits:
            return []

        model = self._load_model()
        pairs = [[query, hit.chunk.text] for hit in hits]
        try:
            raw_scores = model.compute_score(pairs)
        except Exception as exc:
            raise RuntimeError(f"BGE reranker scoring failed: {exc}") from exc
        scores = _normalize_scores(raw_scores)
        ranked = sorted(zip(scores, range(len(hits)), hits), key=lambda item: (item[0], -item[1]), reverse=True)
        return [hit for _, _, hit in ranked[:top_k]]

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model

        try:
            module = importlib.import_module("FlagEmbedding")
            flag_reranker = getattr(module, "FlagReranker")
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("BGE reranker requires the FlagEmbedding package.") from exc

        try:
            self._model = flag_reranker(self.model_name, use_fp16=self.use_fp16)
        except Exception as exc:
            raise RuntimeError(f"BGE reranker model is unavailable: {exc}") from exc
        return self._model


class LocalKeywordReranker:
    """Deterministic fallback used when BGE is unavailable in local development."""

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int = settings.rerank_top_k) -> list[RetrievalHit]:
        query_tokens = _tokens(query)
        scored = [
            (
                _rerank_score(query_tokens, hit),
                index,
                hit,
            )
            for index, hit in enumerate(hits)
        ]
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return [hit for _, _, hit in scored[:top_k]]


class ResilientReranker:
    def __init__(
        self,
        primary: Reranker,
        fallback: Reranker | None = None,
        require_primary: bool = settings.require_external_services,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.require_primary = require_primary

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int = settings.rerank_top_k) -> list[RetrievalHit]:
        try:
            return self.primary.rerank(query, hits, top_k=top_k)
        except RuntimeError:
            if self.require_primary or self.fallback is None:
                raise
            return self.fallback.rerank(query, hits, top_k=top_k)


def create_reranker() -> Reranker:
    provider = settings.rerank_provider.lower()
    if provider in {"local", "keyword"}:
        return LocalKeywordReranker()
    if provider == "bge":
        if not settings.rerank_require_model:
            return LocalKeywordReranker()
        return ResilientReranker(
            primary=BgeReranker(),
            fallback=LocalKeywordReranker(),
            require_primary=settings.rerank_require_model,
        )
    raise RuntimeError(f"Unsupported rerank provider: {settings.rerank_provider}")


def _rerank_score(query_tokens: set[str], hit: RetrievalHit) -> float:
    if not query_tokens:
        return hit.score

    text_tokens = _tokens(hit.chunk.text)
    if not text_tokens:
        return hit.score

    overlap = len(query_tokens & text_tokens) / len(query_tokens)
    return (hit.score * 0.7) + (overlap * 0.3)


def _tokens(text: str) -> set[str]:
    tokens = {token.lower() for token in TOKEN_RE.findall(text)}
    cjk_chars = [char for char in text if "\u4e00" <= char <= "\u9fff"]
    tokens.update(a + b for a, b in zip(cjk_chars, cjk_chars[1:]))
    return tokens


def _normalize_scores(raw_scores: object) -> list[float]:
    if isinstance(raw_scores, int | float):
        return [float(raw_scores)]
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    if not isinstance(raw_scores, Iterable) or isinstance(raw_scores, str):
        raise RuntimeError("BGE reranker returned invalid scores.")
    return [float(score) for score in raw_scores]
