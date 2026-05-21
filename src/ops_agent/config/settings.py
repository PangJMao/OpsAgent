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
    chunk_size: int = 700
    chunk_overlap: int = 120
    embedding_dimensions: int = 256
    min_relevance_score: float = 0.08
    top_k: int = 4
    llm_provider: str = _env("OPS_AGENT_LLM_PROVIDER", "deepseek")
    deepseek_api_key: str = _env("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = _env("DEEPSEEK_MODEL", "deepseek-chat")
    llm_timeout_seconds: float = float(_env("OPS_AGENT_LLM_TIMEOUT_SECONDS", "20"))
    agent_max_retries: int = int(_env("OPS_AGENT_MAX_RETRIES", "2"))


settings = Settings()
