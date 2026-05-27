from __future__ import annotations

import json
import secrets
from pathlib import Path

from ops_agent.context.context_schema import LongTermMemory, LongTermMemoryItem
from ops_agent.models import utc_now_iso


class LongTermMemoryRepository:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path("storage") / "memory" / "long_term_memory.json"

    def load(self, user_id: str) -> LongTermMemory:
        items = [item for item in self._read_items() if item.user_id == user_id and item.is_active]
        return LongTermMemory(
            stable_user_facts=[item.content for item in items if item.memory_type in {"role", "work", "project"}],
            work_preferences=[item.content for item in items if item.memory_type == "preference"],
            frequent_business_scenes=[item.content for item in items if item.memory_type in {"customer", "scene"}],
            saved_constraints=[item.content for item in items if item.memory_type == "constraint"],
            items=items,
        )

    def save_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        *,
        confidence: float = 1.0,
        source: str = "explicit",
        source_message: str = "",
    ) -> LongTermMemoryItem:
        items = self._read_items()
        now = utc_now_iso()
        for item in items:
            if item.user_id == user_id and item.memory_type == memory_type and item.content == content:
                item.updated_at = now
                item.is_active = True
                self._write_items(items)
                return item
        item = LongTermMemoryItem(
            memory_id=secrets.token_hex(12),
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            confidence=confidence,
            source=source,
            source_message=source_message,
            created_at=now,
            updated_at=now,
            is_active=True,
        )
        items.append(item)
        self._write_items(items)
        return item

    def delete_memory(self, memory_id: str) -> None:
        items = [item for item in self._read_items() if item.memory_id != memory_id]
        self._write_items(items)

    def deactivate_memory(self, memory_id: str) -> None:
        items = self._read_items()
        for item in items:
            if item.memory_id == memory_id:
                item.is_active = False
                item.updated_at = utc_now_iso()
        self._write_items(items)

    def _read_items(self) -> list[LongTermMemoryItem]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8") or "[]")
        return [LongTermMemoryItem(**item) for item in raw]

    def _write_items(self, items: list[LongTermMemoryItem]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [item.__dict__ for item in items]
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
