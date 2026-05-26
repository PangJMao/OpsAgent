from fastapi.testclient import TestClient

from ops_agent.config import settings
from ops_agent.main import create_app
from ops_agent.services.auth_service import user_service
from ops_agent.services.conversation_service import conversation_service


def _use_memory_state() -> dict[str, object]:
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
    conversation_service._conversations.clear()
    conversation_service._messages.clear()
    return original


def _restore_settings(original: dict[str, object]) -> None:
    for key, value in original.items():
        object.__setattr__(settings, key, value)


def test_conversations_are_scoped_to_current_user() -> None:
    original = _use_memory_state()
    try:
        root_client = TestClient(create_app())
        root_client.post("/auth/login", json={"username": "root", "password": "root"})
        root_client.post("/users", json={"username": "alice", "password": "password123", "role": "user"})
        root_client.post("/users", json={"username": "bob", "password": "password123", "role": "user"})

        alice = TestClient(create_app())
        bob = TestClient(create_app())
        alice.post("/auth/login", json={"username": "alice", "password": "password123"})
        bob.post("/auth/login", json={"username": "bob", "password": "password123"})

        alice_conversation = alice.post("/conversations", json={"title": "Alice chat"}).json()["conversation"]
        bob_conversation = bob.post("/conversations", json={"title": "Bob chat"}).json()["conversation"]

        alice_list = alice.get("/conversations").json()["conversations"]
        bob_list = bob.get("/conversations").json()["conversations"]
        bob_reads_alice = bob.get(f"/conversations/{alice_conversation['conversation_id']}/messages")
        alice_deletes_bob = alice.delete(f"/conversations/{bob_conversation['conversation_id']}")

        assert [item["title"] for item in alice_list] == ["Alice chat"]
        assert [item["title"] for item in bob_list] == ["Bob chat"]
        assert bob_reads_alice.status_code == 404
        assert alice_deletes_bob.status_code == 404
    finally:
        _restore_settings(original)


def test_default_conversation_title_updates_from_first_question() -> None:
    original = _use_memory_state()
    try:
        client = TestClient(create_app())
        client.post("/auth/login", json={"username": "root", "password": "root"})
        conversation = client.post("/conversations", json={"title": "新对话"}).json()["conversation"]

        conversation_service.add_message(
            "root",
            conversation["conversation_id"],
            "user",
            "D10-D15 阶段可以使用哪些更强的话术？",
        )
        conversations = client.get("/conversations").json()["conversations"]

        assert conversations[0]["title"] == "D10-D15 阶段可以使用哪些更强的话术"
    finally:
        _restore_settings(original)
