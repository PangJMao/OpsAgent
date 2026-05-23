from fastapi.testclient import TestClient

from ops_agent.config import settings
from ops_agent.main import create_app


def test_app_starts_degraded_when_database_is_unavailable() -> None:
    original = {
        "require_external_services": settings.require_external_services,
        "database_url": settings.database_url,
        "vector_provider": settings.vector_provider,
        "root_username": settings.root_username,
        "root_password": settings.root_password,
        "session_secret": settings.session_secret,
    }
    try:
        object.__setattr__(settings, "require_external_services", True)
        object.__setattr__(settings, "database_url", "postgresql://ops_agent:ops_agent@127.0.0.1:1/ops_agent")
        object.__setattr__(settings, "vector_provider", "pgvector")
        object.__setattr__(settings, "root_username", "root")
        object.__setattr__(settings, "root_password", "123456")
        object.__setattr__(settings, "session_secret", "secret")

        with TestClient(create_app()) as client:
            health = client.get("/health")
            login = client.post("/auth/login", json={"username": "root", "password": "123456"})

        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        assert login.status_code == 503
        assert "用户数据库不可用" in login.json()["detail"]
    finally:
        for key, value in original.items():
            object.__setattr__(settings, key, value)
