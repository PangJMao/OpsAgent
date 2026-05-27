from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ops_agent.context.context_schema import UserProfile
from ops_agent.models import utc_now_iso


class UserProfileRepository:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path("storage") / "memory" / "user_profiles.json"

    def load_user_profile(self, user_id: str) -> UserProfile:
        profiles = self._read_profiles()
        payload = profiles.get(user_id)
        if not payload:
            return UserProfile(user_id=user_id)
        return UserProfile(**payload)

    def update_user_profile(self, user_id: str, patch: dict[str, Any]) -> UserProfile:
        profiles = self._read_profiles()
        current = self.load_user_profile(user_id)
        payload = current.__dict__.copy()
        for key, value in patch.items():
            if key not in payload or value in (None, "", []):
                continue
            if isinstance(payload[key], list):
                existing = list(payload[key])
                incoming = value if isinstance(value, list) else [value]
                payload[key] = list(dict.fromkeys([*existing, *incoming]))
            else:
                payload[key] = value
        payload["updated_at"] = utc_now_iso()
        profiles[user_id] = payload
        self._write_profiles(profiles)
        return UserProfile(**payload)

    def _read_profiles(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        return data if isinstance(data, dict) else {}

    def _write_profiles(self, profiles: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
