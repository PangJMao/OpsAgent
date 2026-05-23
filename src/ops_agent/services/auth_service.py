from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import secrets
from typing import Literal

from ops_agent.config import settings
from ops_agent.models import utc_now_iso
from ops_agent.services.database_service import DatabaseService


UserRole = Literal["user", "admin", "root"]


@dataclass
class UserRecord:
    user_id: str
    username: str
    password_hash: str
    role: UserRole
    active: bool = True
    created_at: str = ""


class UserService:
    """User storage facade. PostgreSQL is used when external services are required."""

    def __init__(self) -> None:
        self._users: dict[str, UserRecord] = {}
        self.database = DatabaseService()
        if not settings.require_external_services:
            self._ensure_root_user()

    def authenticate(self, username: str, password: str) -> UserRecord | None:
        user = self.get_by_username(username)
        if user is None or not user.active:
            return None
        if not verify_password(password, user.password_hash):
            return None
        return user

    def create_user(self, username: str, password: str, role: UserRole = "user") -> UserRecord:
        username = username.strip()
        if not username:
            raise ValueError("Username is required.")
        if role == "root":
            raise ValueError("Only the bootstrap root user can have root role.")
        if self.get_by_username(username) is not None:
            raise ValueError("Username already exists.")
        user = UserRecord(
            user_id=secrets.token_hex(12),
            username=username,
            password_hash=hash_password(password),
            role=role,
            created_at=utc_now_iso(),
        )
        if settings.require_external_services:
            with self.database.connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO users (user_id, username, password_hash, role, active)
                        VALUES (%s, %s, %s, %s, true)
                        """,
                        (user.user_id, user.username, user.password_hash, user.role),
                    )
                connection.commit()
            return user
        self._users[user.user_id] = user
        return user

    def delete_user(self, user_id: str) -> None:
        if settings.require_external_services:
            user = self.get(user_id)
            if user is None:
                raise KeyError(user_id)
            if user.role == "root":
                raise ValueError("Root user cannot be deleted.")
            with self.database.connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
                connection.commit()
            return
        user = self._users.get(user_id)
        if user is None:
            raise KeyError(user_id)
        if user.role == "root":
            raise ValueError("Root user cannot be deleted.")
        del self._users[user_id]

    def set_role(self, user_id: str, role: UserRole) -> UserRecord:
        if role == "root":
            raise ValueError("Root role cannot be assigned.")
        if settings.require_external_services:
            user = self.get(user_id)
            if user is None:
                raise KeyError(user_id)
            if user.role == "root":
                raise ValueError("Root user role cannot be changed.")
            with self.database.connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("UPDATE users SET role = %s WHERE user_id = %s", (role, user_id))
                connection.commit()
            updated = self.get(user_id)
            if updated is None:
                raise KeyError(user_id)
            return updated
        user = self._users.get(user_id)
        if user is None:
            raise KeyError(user_id)
        if user.role == "root":
            raise ValueError("Root user role cannot be changed.")
        user.role = role
        return user

    def list_users(self) -> list[UserRecord]:
        if settings.require_external_services:
            with self.database.connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT user_id, username, password_hash, role, active, created_at::text
                        FROM users
                        ORDER BY created_at
                        """
                    )
                    return [_user_from_row(row) for row in cursor.fetchall()]
        return sorted(self._users.values(), key=lambda user: user.created_at)

    def get_by_username(self, username: str) -> UserRecord | None:
        normalized = username.strip()
        if settings.require_external_services:
            with self.database.connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT user_id, username, password_hash, role, active, created_at::text
                        FROM users
                        WHERE username = %s
                        """,
                        (normalized,),
                    )
                    row = cursor.fetchone()
                    return _user_from_row(row) if row else None
        return next((user for user in self._users.values() if user.username == normalized), None)

    def get(self, user_id: str) -> UserRecord | None:
        if settings.require_external_services:
            with self.database.connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT user_id, username, password_hash, role, active, created_at::text
                        FROM users
                        WHERE user_id = %s
                        """,
                        (user_id,),
                    )
                    row = cursor.fetchone()
                    return _user_from_row(row) if row else None
        return self._users.get(user_id)

    def _ensure_root_user(self) -> None:
        username = settings.root_username or "root"
        password = settings.root_password or "root"
        if self.get_by_username(username) is not None:
            return
        if settings.require_external_services:
            user = UserRecord(
                user_id="root",
                username=username,
                password_hash=hash_password(password),
                role="root",
                created_at=utc_now_iso(),
            )
            with self.database.connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO users (user_id, username, password_hash, role, active)
                        VALUES (%s, %s, %s, 'root', true)
                        ON CONFLICT (user_id) DO UPDATE SET
                            username = excluded.username,
                            password_hash = excluded.password_hash,
                            role = 'root',
                            active = true
                        """,
                        (user.user_id, user.username, user.password_hash),
                    )
                connection.commit()
            return
        user = UserRecord(
            user_id="root",
            username=username,
            password_hash=hash_password(password),
            role="root",
            created_at=utc_now_iso(),
        )
        self._users[user.user_id] = user


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return f"pbkdf2_sha256${salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, salt, digest = stored_hash.split("$", 2)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return hmac.compare_digest(candidate.hex(), digest)


def _user_from_row(row) -> UserRecord:
    return UserRecord(
        user_id=row[0],
        username=row[1],
        password_hash=row[2],
        role=row[3],
        active=bool(row[4]),
        created_at=str(row[5]),
    )


user_service = UserService()
