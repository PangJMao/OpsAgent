from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ops_agent.api.security import public_user, require_admin_user
from ops_agent.models import CreateUserRequest, UpdateUserRoleRequest
from ops_agent.services.auth_service import UserRole, user_service
from ops_agent.services.database_service import StartupConfigurationError

router = APIRouter(prefix="/users", tags=["users"])


@router.get("")
def list_users(request: Request) -> dict[str, object]:
    require_admin_user(request)
    try:
        return {"users": [public_user(user) for user in user_service.list_users()]}
    except StartupConfigurationError as exc:
        raise HTTPException(status_code=503, detail=f"用户数据库不可用：{exc}") from exc


@router.post("")
def create_user(request: Request, payload: CreateUserRequest) -> dict[str, object]:
    actor = require_admin_user(request)
    role = _normalize_role(payload.role)
    if actor.role != "root" and role == "admin":
        raise HTTPException(status_code=403, detail="Only root can create admin users.")
    try:
        user = user_service.create_user(payload.username, payload.password, role=role)
    except StartupConfigurationError as exc:
        raise HTTPException(status_code=503, detail=f"用户数据库不可用：{exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"user": public_user(user)}


@router.patch("/{user_id}/role")
def update_user_role(user_id: str, request: Request, payload: UpdateUserRoleRequest) -> dict[str, object]:
    actor = require_admin_user(request)
    if actor.role != "root":
        raise HTTPException(status_code=403, detail="Only root can change admin privileges.")
    try:
        user = user_service.set_role(user_id, _normalize_role(payload.role))
    except StartupConfigurationError as exc:
        raise HTTPException(status_code=503, detail=f"用户数据库不可用：{exc}") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="User not found.") from exc
    except StartupConfigurationError as exc:
        raise HTTPException(status_code=503, detail=f"用户数据库不可用：{exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"user": public_user(user)}


@router.delete("/{user_id}")
def delete_user(user_id: str, request: Request) -> dict[str, object]:
    actor = require_admin_user(request)
    target = user_service.get(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found.")
    if actor.role != "root" and target.role == "admin":
        raise HTTPException(status_code=403, detail="Only root can delete admin users.")
    try:
        user_service.delete_user(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


def _normalize_role(role: str) -> UserRole:
    if role == "admin":
        return "admin"
    if role == "user":
        return "user"
    raise HTTPException(status_code=400, detail="Role must be user or admin.")
