from pathlib import Path

from ops_agent.config.settings import _env, _load_dotenv


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
