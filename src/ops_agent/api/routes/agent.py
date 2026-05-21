from __future__ import annotations

from fastapi import APIRouter

from ops_agent.models import AgentRunRequest
from ops_agent.services import AgentRunCommand, AgentService, agent_answer_to_dict

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/run")
def run_agent(request: AgentRunRequest) -> dict[str, object]:
    service = AgentService()
    answer = service.run(AgentRunCommand(question=request.question))
    return agent_answer_to_dict(answer)
