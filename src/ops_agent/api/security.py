from __future__ import annotations

from fastapi import Header, HTTPException, Request

from ops_agent.services.auth_service import UserRecord, user_service
from ops_agent.services.permission_service import PermissionContext, context_from_headers


def current_context(
    request: Request,
    x_user_id: str | None = Header(default=None),
    x_role: str | None = Header(default=None),
    x_scopes: str | None = Header(default=None),
) -> PermissionContext:
    session_user_id = request.session.get("user_id") if hasattr(request, "session") else None
    if session_user_id:
        user = user_service.get(str(session_user_id))
        if user and user.active:
            return PermissionContext(user_id=user.user_id, role=user.role, knowledge_scopes=("default",))
    return context_from_headers(x_user_id, x_role, x_scopes)


def require_login(request: Request) -> UserRecord:
    user_id = request.session.get("user_id") if hasattr(request, "session") else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Login required.")
    user = user_service.get(str(user_id))
    if user is None or not user.active:
        request.session.clear()
        raise HTTPException(status_code=401, detail="Login required.")
    return user


def require_admin_user(request: Request) -> UserRecord:
    user = require_login(request)
    if user.role not in {"admin", "root"}:
        raise HTTPException(status_code=403, detail="Admin role required.")
    return user


def public_user(user: UserRecord) -> dict[str, object]:
    return {
        "user_id": user.user_id,
        "username": user.username,
        "role": user.role,
        "active": user.active,
        "created_at": user.created_at,
    }
