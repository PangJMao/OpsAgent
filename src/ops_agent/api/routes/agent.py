from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request

from ops_agent.api.security import current_context
from ops_agent.models import AgentRunRequest
from ops_agent.services import AgentRunCommand, AgentService, agent_answer_to_dict
from ops_agent.services.database_service import StartupConfigurationError
from ops_agent.services.permission_service import PermissionService
from ops_agent.services.trace_service import TraceRecorder

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/run")
def run_agent(
    request: AgentRunRequest,
    http_request: Request,
    x_user_id: str | None = Header(default=None),
    x_role: str | None = Header(default=None),
    x_scopes: str | None = Header(default=None),
) -> dict[str, object]:
    context = current_context(http_request, x_user_id, x_role, x_scopes)
    try:
        PermissionService().require(context, "agent.run")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    recorder = TraceRecorder()
    recorder.set_context(
        user_id=context.user_id,
        role=context.role,
        knowledge_scopes=list(context.knowledge_scopes),
    )
    service = AgentService(recorder=recorder)
    try:
        answer = service.run(AgentRunCommand(question=request.question))
        return agent_answer_to_dict(answer)
    except (RuntimeError, StartupConfigurationError) as exc:
        raise HTTPException(status_code=503, detail=f"Agent 工作流暂不可用：{exc}") from exc
