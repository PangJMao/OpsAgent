from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import socket
from typing import Iterator
from urllib.parse import urlparse

from ops_agent.config import settings


class StartupConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatabaseStatus:
    configured: bool
    connected: bool
    vector_ready: bool
    message: str


class DatabaseService:
    """PostgreSQL initialization for users and pgvector-backed knowledge chunks."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = settings.database_url if database_url is None else database_url
        self.connect_timeout_seconds = 1

    def validate_startup(self) -> DatabaseStatus:
        config_errors = settings.startup_errors()
        if config_errors:
            raise StartupConfigurationError("; ".join(config_errors))
        if not settings.require_external_services:
            return DatabaseStatus(
                configured=bool(self.database_url),
                connected=False,
                vector_ready=False,
                message="External services are optional in this environment.",
            )
        self._preflight_tcp()
        try:
            with self.connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                    vector_ready = bool(cursor.fetchone()[0])
        except Exception as exc:
            raise StartupConfigurationError(f"Database is unavailable: {exc}") from exc
        if not vector_ready:
            raise StartupConfigurationError("pgvector extension is not available in the database.")
        return DatabaseStatus(
            configured=True,
            connected=True,
            vector_ready=True,
            message="Database and pgvector are ready.",
        )

    @contextmanager
    def connect(self) -> Iterator:
        try:
            import psycopg
        except ImportError as exc:
            raise StartupConfigurationError("psycopg is required for PostgreSQL access.") from exc
        if not self.database_url:
            raise StartupConfigurationError("OPS_AGENT_DATABASE_URL is not configured.")
        self._preflight_tcp()
        try:
            with psycopg.connect(self.database_url, connect_timeout=self.connect_timeout_seconds) as connection:
                yield connection
        except Exception as exc:
            raise StartupConfigurationError(f"Database connection failed: {exc}") from exc

    def _preflight_tcp(self) -> None:
        parsed = urlparse(self.database_url)
        host = parsed.hostname
        port = parsed.port or 5432
        if not host:
            raise StartupConfigurationError("OPS_AGENT_DATABASE_URL must include a database host.")
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError as exc:
            raise StartupConfigurationError(f"Database host is unreachable: {host}:{port}") from exc

    def initialize(self) -> None:
        if not settings.require_external_services:
            return
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user', 'admin', 'root')),
                        active BOOLEAN NOT NULL DEFAULT true,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS knowledge_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        text TEXT NOT NULL,
                        start_char INTEGER NOT NULL,
                        end_char INTEGER NOT NULL,
                        metadata_json JSONB NOT NULL,
                        embedding vector({settings.embedding_dimensions}) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_document ON knowledge_chunks(document_id)")
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        conversation_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                        title TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        message_id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                        user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        citations_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at DESC)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON conversation_messages(conversation_id, created_at)"
                )
                from ops_agent.services.auth_service import hash_password

                cursor.execute(
                    """
                    INSERT INTO users (user_id, username, password_hash, role, active)
                    VALUES ('root', %s, %s, 'root', true)
                    ON CONFLICT (user_id) DO UPDATE SET
                        username = excluded.username,
                        password_hash = excluded.password_hash,
                        role = 'root',
                        active = true
                    """,
                    (settings.root_username, hash_password(settings.root_password)),
                )
            connection.commit()
