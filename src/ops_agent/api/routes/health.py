from __future__ import annotations

from fastapi import APIRouter

from ops_agent.config import settings
from ops_agent.services.runtime_state import runtime_state

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    storage_checks = {
        "storage_dir": settings.storage_dir.exists(),
        "documents_dir": settings.documents_dir.exists(),
        "indexes_dir": settings.indexes_dir.exists(),
        "traces_dir": settings.traces_dir.exists(),
    }
    storage_ok = all(storage_checks.values())
    runtime = runtime_state.snapshot()
    status = "ok" if storage_ok and runtime["status"] == "ok" else "degraded"
    return {
        "status": status,
        "runtime": runtime,
        "storage": storage_checks,
        "llm_provider": settings.llm_provider,
        "llm_configured": bool(settings.deepseek_api_key),
        "top_k": settings.top_k,
        "min_relevance_score": settings.min_relevance_score,
    }
