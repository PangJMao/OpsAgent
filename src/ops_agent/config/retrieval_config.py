from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    """Hybrid Search 参数集中配置，避免权重和阈值散落在检索代码里。"""

    rewritten_query_limit: int = 6
    per_query_top_k: int = 10
    hybrid_pool_size: int = 30
    rerank_top_k: int = 8
    final_top_k: int = 6
    max_per_sheet: int = 3
    mmr_lambda: float = 0.7
    high_confidence_threshold: float = 0.75
    medium_confidence_threshold: float = 0.60
    vector_weight: float = 0.45
    keyword_weight: float = 0.30
    metadata_weight: float = 0.15
    source_bonus_weight: float = 0.10
    exact_vector_weight: float = 0.35
    exact_keyword_weight: float = 0.40


DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig()
