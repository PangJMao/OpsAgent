from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ops_agent.api.security import public_user, require_login
from ops_agent.models import LoginRequest
from ops_agent.services.database_service import StartupConfigurationError
from ops_agent.services.auth_service import user_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
def login(request: Request, payload: LoginRequest) -> dict[str, object]:
    try:
        user = user_service.authenticate(payload.username, payload.password)
    except StartupConfigurationError as exc:
        raise HTTPException(status_code=503, detail=f"用户数据库不可用：{exc}") from exc
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    request.session["user_id"] = user.user_id
    return {"user": public_user(user)}


@router.post("/logout")
def logout(request: Request) -> dict[str, object]:
    request.session.clear()
    return {"ok": True}


@router.get("/me")
def me(request: Request) -> dict[str, object]:
    return {"user": public_user(require_login(request))}
