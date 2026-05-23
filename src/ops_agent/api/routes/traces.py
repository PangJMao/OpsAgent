from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Query

from ops_agent.services.permission_service import PermissionService, context_from_headers
from ops_agent.services.trace_service import TraceStore

router = APIRouter(prefix="/traces", tags=["traces"])


@router.get("")
def list_traces(
    limit: int = Query(default=50, ge=1, le=200),
    x_user_id: str | None = Header(default=None),
    x_role: str | None = Header(default=None),
    x_scopes: str | None = Header(default=None),
) -> dict[str, object]:
    _require_trace_read(x_user_id, x_role, x_scopes)
    return {"traces": TraceStore().list(limit=limit)}


@router.get("/{trace_id}")
def get_trace(
    trace_id: str,
    x_user_id: str | None = Header(default=None),
    x_role: str | None = Header(default=None),
    x_scopes: str | None = Header(default=None),
) -> dict[str, object]:
    _require_trace_read(x_user_id, x_role, x_scopes)
    try:
        return TraceStore().get(trace_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}") from exc


def _require_trace_read(user_id: str | None, role: str | None, scopes: str | None) -> None:
    context = context_from_headers(user_id, role, scopes)
    try:
        PermissionService().require(context, "trace.read")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
