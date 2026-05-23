from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import secrets
from typing import Any

from ops_agent.config import settings
from ops_agent.models import utc_now_iso
from ops_agent.services.database_service import DatabaseService


@dataclass
class ConversationRecord:
    conversation_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str


@dataclass
class MessageRecord:
    message_id: str
    conversation_id: str
    user_id: str
    role: str
    content: str
    citations: list[dict[str, Any]]
    created_at: str


class ConversationService:
    def __init__(self) -> None:
        self._conversations: dict[str, ConversationRecord] = {}
        self._messages: dict[str, list[MessageRecord]] = {}

    def list_conversations(self, user_id: str) -> list[dict[str, Any]]:
        if settings.require_external_services:
            with DatabaseService().connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT conversation_id, user_id, title, created_at::text, updated_at::text
                        FROM conversations
                        WHERE user_id = %s
                        ORDER BY updated_at DESC
                        """,
                        (user_id,),
                    )
                    return [asdict(_conversation_from_row(row)) for row in cursor.fetchall()]
        records = [record for record in self._conversations.values() if record.user_id == user_id]
        records.sort(key=lambda record: record.updated_at, reverse=True)
        return [asdict(record) for record in records]

    def create_conversation(self, user_id: str, title: str = "新对话") -> dict[str, Any]:
        now = utc_now_iso()
        record = ConversationRecord(
            conversation_id=secrets.token_hex(12),
            user_id=user_id,
            title=(title or "新对话").strip()[:80],
            created_at=now,
            updated_at=now,
        )
        if settings.require_external_services:
            with DatabaseService().connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO conversations (conversation_id, user_id, title)
                        VALUES (%s, %s, %s)
                        """,
                        (record.conversation_id, record.user_id, record.title),
                    )
                connection.commit()
            return asdict(record)
        self._conversations[record.conversation_id] = record
        self._messages[record.conversation_id] = []
        return asdict(record)

    def delete_conversation(self, user_id: str, conversation_id: str) -> None:
        self._require_owner(user_id, conversation_id)
        if settings.require_external_services:
            with DatabaseService().connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM conversations WHERE conversation_id = %s AND user_id = %s",
                        (conversation_id, user_id),
                    )
                connection.commit()
            return
        self._conversations.pop(conversation_id, None)
        self._messages.pop(conversation_id, None)

    def list_messages(self, user_id: str, conversation_id: str) -> list[dict[str, Any]]:
        self._require_owner(user_id, conversation_id)
        if settings.require_external_services:
            with DatabaseService().connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT message_id, conversation_id, user_id, role, content, citations_json, created_at::text
                        FROM conversation_messages
                        WHERE conversation_id = %s AND user_id = %s
                        ORDER BY created_at
                        """,
                        (conversation_id, user_id),
                    )
                    return [asdict(_message_from_row(row)) for row in cursor.fetchall()]
        return [asdict(message) for message in self._messages.get(conversation_id, [])]

    def add_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self._require_owner(user_id, conversation_id)
        now = utc_now_iso()
        message = MessageRecord(
            message_id=secrets.token_hex(12),
            conversation_id=conversation_id,
            user_id=user_id,
            role=role,
            content=content,
            citations=citations or [],
            created_at=now,
        )
        if settings.require_external_services:
            with DatabaseService().connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO conversation_messages (
                            message_id, conversation_id, user_id, role, content, citations_json
                        )
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            message.message_id,
                            message.conversation_id,
                            message.user_id,
                            message.role,
                            message.content,
                            json.dumps(message.citations, ensure_ascii=False),
                        ),
                    )
                    cursor.execute(
                        "UPDATE conversations SET updated_at = now() WHERE conversation_id = %s",
                        (conversation_id,),
                    )
                connection.commit()
            return asdict(message)
        self._messages.setdefault(conversation_id, []).append(message)
        conversation = self._conversations[conversation_id]
        conversation.updated_at = now
        return asdict(message)

    def _require_owner(self, user_id: str, conversation_id: str) -> None:
        if settings.require_external_services:
            with DatabaseService().connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT 1 FROM conversations WHERE conversation_id = %s AND user_id = %s",
                        (conversation_id, user_id),
                    )
                    if cursor.fetchone() is None:
                        raise KeyError(conversation_id)
            return
        record = self._conversations.get(conversation_id)
        if record is None or record.user_id != user_id:
            raise KeyError(conversation_id)


def _conversation_from_row(row) -> ConversationRecord:
    return ConversationRecord(
        conversation_id=row[0],
        user_id=row[1],
        title=row[2],
        created_at=str(row[3]),
        updated_at=str(row[4]),
    )


def _message_from_row(row) -> MessageRecord:
    citations = row[5] if isinstance(row[5], list) else json.loads(row[5] or "[]")
    return MessageRecord(
        message_id=row[0],
        conversation_id=row[1],
        user_id=row[2],
        role=row[3],
        content=row[4],
        citations=citations,
        created_at=str(row[6]),
    )


conversation_service = ConversationService()
