from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Query, Request

from ops_agent.api.security import current_context
from ops_agent.config import settings
from ops_agent.models import AskRequest, IngestRequest
from ops_agent.services import RagService, answer_to_dict
from ops_agent.services.database_service import StartupConfigurationError
from ops_agent.services.permission_service import PermissionService
from ops_agent.services.trace_service import TraceRecorder

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ingest")
def ingest_document(
    request: IngestRequest,
    http_request: Request,
    x_user_id: str | None = Header(default=None),
    x_role: str | None = Header(default=None),
    x_scopes: str | None = Header(default=None),
) -> dict[str, object]:
    context = current_context(http_request, x_user_id, x_role, x_scopes)
    _require(context, "rag.ingest")
    recorder = _recorder_for_context(context)
    service = RagService(recorder=recorder)
    try:
        return service.ingest(Path(request.path))
    except (OSError, RuntimeError, StartupConfigurationError) as exc:
        raise HTTPException(status_code=503, detail=f"知识库入库失败：{exc}") from exc


@router.post("/ask")
def ask_question(
    request: AskRequest,
    http_request: Request,
    x_user_id: str | None = Header(default=None),
    x_role: str | None = Header(default=None),
    x_scopes: str | None = Header(default=None),
) -> dict[str, object]:
    context = current_context(http_request, x_user_id, x_role, x_scopes)
    _require(context, "rag.ask")
    recorder = _recorder_for_context(context)
    service = RagService(recorder=recorder)
    try:
        return answer_to_dict(service.ask(request.question))
    except (RuntimeError, StartupConfigurationError) as exc:
        raise HTTPException(status_code=503, detail=f"知识库检索不可用：{exc}") from exc


@router.post("/documents")
async def upload_document(
    http_request: Request,
    filename: str = Query(..., min_length=1),
    x_user_id: str | None = Header(default=None),
    x_role: str | None = Header(default=None),
    x_scopes: str | None = Header(default=None),
) -> dict[str, object]:
    context = current_context(http_request, x_user_id, x_role, x_scopes)
    _require(context, "rag.ingest")
    content = await http_request.body()
    if not content:
        raise HTTPException(status_code=400, detail="Document content is required.")
    target = settings.documents_dir / Path(filename).name
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    recorder = _recorder_for_context(context)
    try:
        return RagService(recorder=recorder).ingest(target)
    except (OSError, RuntimeError, StartupConfigurationError) as exc:
        raise HTTPException(status_code=503, detail=f"文档入库失败：{exc}") from exc


def _require(context, action: str) -> None:
    try:
        PermissionService().require(context, action)  # type: ignore[arg-type]
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _recorder_for_context(context) -> TraceRecorder:
    recorder = TraceRecorder()
    recorder.set_context(
        user_id=context.user_id,
        role=context.role,
        knowledge_scopes=list(context.knowledge_scopes),
    )
    return recorder
