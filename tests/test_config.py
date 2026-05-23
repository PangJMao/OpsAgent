from pathlib import Path

from ops_agent.config.settings import _env, _load_dotenv
from ops_agent.services.database_service import DatabaseService, StartupConfigurationError


def test_load_dotenv_reads_key_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
        # local config
        DEEPSEEK_API_KEY="test-key"
        DEEPSEEK_MODEL=deepseek-chat
        """,
        encoding="utf-8",
    )

    values = _load_dotenv(env_file)

    assert values["DEEPSEEK_API_KEY"] == "test-key"
    assert values["DEEPSEEK_MODEL"] == "deepseek-chat"


def test_env_prefers_process_environment(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "process-key")

    assert _env("DEEPSEEK_API_KEY", "fallback") == "process-key"


def test_database_startup_reports_missing_database_url() -> None:
    from ops_agent.config import settings

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
        object.__setattr__(settings, "database_url", "")
        object.__setattr__(settings, "vector_provider", "pgvector")
        object.__setattr__(settings, "root_username", "root")
        object.__setattr__(settings, "root_password", "password")
        object.__setattr__(settings, "session_secret", "secret")
        try:
            DatabaseService(database_url="").validate_startup()
        except StartupConfigurationError as exc:
            assert "OPS_AGENT_DATABASE_URL" in str(exc)
        else:
            raise AssertionError("Expected StartupConfigurationError")
    finally:
        for key, value in original.items():
            object.__setattr__(settings, key, value)
