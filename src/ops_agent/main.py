from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from ops_agent.api.routes import (
    agent_router,
    auth_router,
    conversations_router,
    evaluation_router,
    health_router,
    rag_router,
    tasks_router,
    traces_router,
    users_router,
)
from ops_agent.config import settings
from ops_agent.services.database_service import DatabaseService, StartupConfigurationError
from ops_agent.services.runtime_state import runtime_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime_state.status = "starting"
    runtime_state.startup_errors.clear()
    runtime_state.components.clear()
    try:
        database_status = DatabaseService().validate_startup()
        DatabaseService().initialize()
        runtime_state.mark_ready("database", database_status.__dict__)
    except StartupConfigurationError as exc:
        runtime_state.mark_degraded("database", str(exc))
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="OpsAgent API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret or "dev-session-secret",
        same_site="lax",
        https_only=False,
    )

    @app.middleware("http")
    async def prevent_frontend_cache(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(conversations_router)
    app.include_router(users_router)
    app.include_router(rag_router)
    app.include_router(agent_router)
    app.include_router(evaluation_router)
    app.include_router(tasks_router)
    app.include_router(traces_router)
    web_dir = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "ok": False,
                "error": "validation_error",
                "message": "请求参数格式不正确。",
                "details": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": "internal_error",
                "message": "系统处理请求时遇到异常，请稍后重试或联系管理员查看 trace。",
                "details": str(exc),
            },
        )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    return app


app = create_app()
