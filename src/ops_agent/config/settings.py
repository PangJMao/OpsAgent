from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOTENV_VALUES: dict[str, str] = {}


def _load_dotenv(path: Path = PROJECT_ROOT / ".env") -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _env(name: str, default: str = "") -> str:
    # 系统环境变量优先，方便线上部署覆盖本地 .env 配置。
    return os.getenv(name, DOTENV_VALUES.get(name, default))


DOTENV_VALUES = _load_dotenv()


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    storage_dir: Path = project_root / "storage"
    documents_dir: Path = storage_dir / "documents"
    normalized_dir: Path = storage_dir / "normalized"
    indexes_dir: Path = storage_dir / "indexes"
    traces_dir: Path = storage_dir / "traces"
    vector_store_path: Path = indexes_dir / "rag_index.db"
    require_external_services: bool = _env("OPS_AGENT_REQUIRE_EXTERNAL_SERVICES", "false").lower() == "true"
    database_url: str = _env("OPS_AGENT_DATABASE_URL", "")
    vector_provider: str = _env("OPS_AGENT_VECTOR_PROVIDER", "local")
    root_username: str = _env("OPS_AGENT_ROOT_USERNAME", "")
    root_password: str = _env("OPS_AGENT_ROOT_PASSWORD", "")
    session_secret: str = _env("OPS_AGENT_SESSION_SECRET", "")
    chunk_size: int = int(_env("OPS_AGENT_CHUNK_SIZE", "900"))
    chunk_overlap: int = int(_env("OPS_AGENT_CHUNK_OVERLAP", "150"))
    embedding_dimensions: int = 256
    embedding_provider: str = _env("OPS_AGENT_EMBEDDING_PROVIDER", "hashing")
    embedding_api_key: str = _env("OPS_AGENT_EMBEDDING_API_KEY", _env("OPENAI_API_KEY", ""))
    embedding_base_url: str = _env("OPS_AGENT_EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    embedding_model: str = _env("OPS_AGENT_EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_timeout_seconds: float = float(_env("OPS_AGENT_EMBEDDING_TIMEOUT_SECONDS", "20"))
    document_vector_provider: str = _env("OPS_AGENT_DOCUMENT_VECTOR_PROVIDER", _env("OPS_AGENT_VECTOR_PROVIDER", "pgvector"))
    pdf_parser_backend: str = _env("OPS_AGENT_PDF_PARSER_BACKEND", "auto")
    retrieval_mode: str = _env("OPS_AGENT_RETRIEVAL_MODE", "hybrid")
    hybrid_vector_weight: float = float(_env("OPS_AGENT_HYBRID_VECTOR_WEIGHT", "0.65"))
    hybrid_bm25_weight: float = float(_env("OPS_AGENT_HYBRID_BM25_WEIGHT", "0.35"))
    min_relevance_score: float = 0.08
    top_k: int = 4
    rerank_provider: str = _env("OPS_AGENT_RERANK_PROVIDER", "local")
    rerank_model: str = _env("OPS_AGENT_RERANK_MODEL", "BAAI/bge-reranker-base")
    rerank_use_fp16: bool = _env("OPS_AGENT_RERANK_USE_FP16", "true").lower() == "true"
    rerank_require_model: bool = _env("OPS_AGENT_RERANK_REQUIRE_MODEL", "false").lower() == "true"
    retrieval_top_k: int = int(_env("OPS_AGENT_RETRIEVAL_TOP_K", "12"))
    rerank_top_k: int = int(_env("OPS_AGENT_RERANK_TOP_K", "3"))
    llm_provider: str = _env("OPS_AGENT_LLM_PROVIDER", "deepseek")
    deepseek_api_key: str = _env("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = _env("DEEPSEEK_MODEL", "deepseek-chat")
    llm_timeout_seconds: float = float(_env("OPS_AGENT_LLM_TIMEOUT_SECONDS", "20"))
    agent_max_retries: int = int(_env("OPS_AGENT_MAX_RETRIES", "2"))

    def startup_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.require_external_services:
            return errors
        if not self.database_url:
            errors.append("OPS_AGENT_DATABASE_URL is required.")
        if self.vector_provider != "pgvector":
            errors.append("OPS_AGENT_VECTOR_PROVIDER must be pgvector.")
        if self.embedding_provider in {"openai", "openai-compatible", "remote", "real"} and not self.embedding_api_key:
            errors.append("OPS_AGENT_EMBEDDING_API_KEY or OPENAI_API_KEY is required.")
        if self.rerank_provider not in {"bge", "local", "keyword"}:
            errors.append("OPS_AGENT_RERANK_PROVIDER must be bge, local, or keyword.")
        if not self.root_username:
            errors.append("OPS_AGENT_ROOT_USERNAME is required.")
        if not self.root_password:
            errors.append("OPS_AGENT_ROOT_PASSWORD is required.")
        if not self.session_secret:
            errors.append("OPS_AGENT_SESSION_SECRET is required.")
        return errors


settings = Settings()
