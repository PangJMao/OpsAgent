from __future__ import annotations

from fastapi import FastAPI

from ops_agent.api.routes import agent_router, health_router, rag_router


def create_app() -> FastAPI:
    app = FastAPI(title="OpsAgent API", version="0.1.0")
    app.include_router(health_router)
    app.include_router(rag_router)
    app.include_router(agent_router)
    return app


app = create_app()
