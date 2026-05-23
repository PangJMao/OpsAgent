from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException

from ops_agent.models import EvaluationRunRequest
from ops_agent.services.evaluation_service import EvaluationCase, EvaluationService
from ops_agent.services.permission_service import PermissionService, context_from_headers
from ops_agent.services.task_queue import task_queue

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run")
def run_evaluation(
    request: EvaluationRunRequest,
    x_user_id: str | None = Header(default=None),
    x_role: str | None = Header(default=None),
    x_scopes: str | None = Header(default=None),
) -> dict[str, object]:
    context = context_from_headers(x_user_id, x_role, x_scopes)
    try:
        PermissionService().require(context, "evaluation.run")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    cases = [
        EvaluationCase(
            question=case.question,
            expected_answer_contains=case.expected_answer_contains,
            expect_refused=case.expect_refused,
            require_citation=case.require_citation,
        )
        for case in request.cases
    ]
    record = task_queue.submit("evaluation.run", lambda: EvaluationService().run(cases))
    return {
        "task_id": record.task_id,
        "status": record.status,
        "result": record.result,
        "error": record.error,
    }
