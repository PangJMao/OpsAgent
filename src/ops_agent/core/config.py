from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """本地 RAG 最小可用版本的运行配置。

    当前默认值让第一阶段保持自包含。后续接入 pgvector 时，只需要替换向量存储实现，
    不需要重写 RAG 编排流程。
    """

    project_root: Path = Path(__file__).resolve().parents[3]
    storage_dir: Path = project_root / "storage"
    documents_dir: Path = storage_dir / "documents"
    normalized_dir: Path = storage_dir / "normalized"
    indexes_dir: Path = storage_dir / "indexes"
    traces_dir: Path = storage_dir / "traces"
    vector_store_path: Path = indexes_dir / "rag_index.db"
    chunk_size: int = 700
    chunk_overlap: int = 120
    embedding_dimensions: int = 256
    min_relevance_score: float = 0.08
    top_k: int = 4


settings = Settings()
