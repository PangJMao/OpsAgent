from fastapi.testclient import TestClient

from ops_agent.config import settings
from ops_agent.main import create_app
from ops_agent.services.auth_service import user_service


def _use_memory_users() -> dict[str, object]:
    original = {
        "require_external_services": settings.require_external_services,
        "root_username": settings.root_username,
        "root_password": settings.root_password,
    }
    object.__setattr__(settings, "require_external_services", False)
    object.__setattr__(settings, "root_username", "root")
    object.__setattr__(settings, "root_password", "root")
    user_service._users.clear()
    user_service._ensure_root_user()
    return original


def _restore_settings(original: dict[str, object]) -> None:
    for key, value in original.items():
        object.__setattr__(settings, key, value)


def test_root_can_login_and_manage_users() -> None:
    original = _use_memory_users()
    try:
        client = TestClient(create_app())

        login = client.post("/auth/login", json={"username": "root", "password": "root"})
        assert login.status_code == 200
        assert login.json()["user"]["role"] == "root"

        created = client.post(
            "/users",
            json={"username": "alice", "password": "password123", "role": "user"},
        )
        assert created.status_code == 200
        user_id = created.json()["user"]["user_id"]

        promoted = client.patch(f"/users/{user_id}/role", json={"role": "admin"})
        assert promoted.status_code == 200
        assert promoted.json()["user"]["role"] == "admin"

        deleted = client.delete(f"/users/{user_id}")
        assert deleted.status_code == 200
    finally:
        _restore_settings(original)


def test_common_user_cannot_access_user_admin() -> None:
    original = _use_memory_users()
    try:
        root_client = TestClient(create_app())
        root_client.post("/auth/login", json={"username": "root", "password": "root"})
        root_client.post("/users", json={"username": "bob", "password": "password123", "role": "user"})

        user_client = TestClient(create_app())
        assert user_client.post("/auth/login", json={"username": "bob", "password": "password123"}).status_code == 200

        response = user_client.get("/users")

        assert response.status_code == 403
    finally:
        _restore_settings(original)


def test_admin_can_only_manage_lower_privilege_users() -> None:
    original = _use_memory_users()
    try:
        root_client = TestClient(create_app())
        root_client.post("/auth/login", json={"username": "root", "password": "root"})
        admin = root_client.post(
            "/users",
            json={"username": "manager", "password": "password123", "role": "admin"},
        ).json()["user"]
        peer = root_client.post(
            "/users",
            json={"username": "peer", "password": "password123", "role": "admin"},
        ).json()["user"]

        admin_client = TestClient(create_app())
        assert (
            admin_client.post("/auth/login", json={"username": "manager", "password": "password123"}).status_code
            == 200
        )

        created_user = admin_client.post(
            "/users",
            json={"username": "operator", "password": "password123", "role": "user"},
        )
        created_admin = admin_client.post(
            "/users",
            json={"username": "new-admin", "password": "password123", "role": "admin"},
        )
        delete_peer = admin_client.delete(f"/users/{peer['user_id']}")
        delete_self = admin_client.delete(f"/users/{admin['user_id']}")

        assert created_user.status_code == 200
        assert created_admin.status_code == 403
        assert delete_peer.status_code == 403
        assert delete_self.status_code == 403
    finally:
        _restore_settings(original)


def test_frontend_index_is_served() -> None:
    original = _use_memory_users()
    try:
        client = TestClient(create_app())

        response = client.get("/")

        assert response.status_code == 200
        assert "企业知识库工作台" in response.text
    finally:
        _restore_settings(original)
