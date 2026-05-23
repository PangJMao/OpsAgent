from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Header, HTTPException

from ops_agent.services.permission_service import PermissionService, context_from_headers
from ops_agent.services.task_queue import task_queue

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("")
def list_tasks(
    x_user_id: str | None = Header(default=None),
    x_role: str | None = Header(default=None),
    x_scopes: str | None = Header(default=None),
) -> dict[str, object]:
    context = context_from_headers(x_user_id, x_role, x_scopes)
    try:
        PermissionService().require(context, "task.read")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"tasks": [asdict(task) for task in task_queue.list()]}


@router.get("/{task_id}")
def get_task(
    task_id: str,
    x_user_id: str | None = Header(default=None),
    x_role: str | None = Header(default=None),
    x_scopes: str | None = Header(default=None),
) -> dict[str, object]:
    context = context_from_headers(x_user_id, x_role, x_scopes)
    try:
        PermissionService().require(context, "task.read")
        return asdict(task_queue.get(task_id))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
