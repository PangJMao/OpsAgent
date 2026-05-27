from __future__ import annotations

import re

from ops_agent.context.context_schema import AgentContext
from ops_agent.context.long_term_memory import LongTermMemoryRepository
from ops_agent.context.short_term_memory import ShortTermMemoryStore, short_term_memory_store
from ops_agent.context.user_profile import UserProfileRepository


class MemoryUpdater:
    def __init__(
        self,
        short_term: ShortTermMemoryStore | None = None,
        long_term: LongTermMemoryRepository | None = None,
        profiles: UserProfileRepository | None = None,
    ) -> None:
        self.short_term = short_term or short_term_memory_store
        self.long_term = long_term or LongTermMemoryRepository()
        self.profiles = profiles or UserProfileRepository()

    def update_after_response(
        self,
        agent_context: AgentContext,
        answer: str,
        metadata: dict,
    ) -> None:
        user_id = agent_context.user_id
        session_id = agent_context.session_id
        message = agent_context.current_message
        self.short_term.append_message(session_id, "user", message)
        self.short_term.append_message(session_id, "assistant", answer, metadata)
        self.short_term.update_state(
            session_id,
            agent_context.resolved_message or message,
            intent=str(metadata.get("intent") or ""),
            decision=str(metadata.get("decision") or ""),
            sources=list(metadata.get("sources") or []),
        )
        explicit = extract_explicit_memory(message)
        for item in explicit:
            self.long_term.save_memory(
                user_id,
                item["memory_type"],
                item["content"],
                confidence=1.0,
                source="explicit",
                source_message=message,
            )
        profile_patch = profile_patch_from_memory(explicit)
        if profile_patch:
            self.profiles.update_user_profile(user_id, profile_patch)


def extract_explicit_memory(message: str) -> list[dict[str, str]]:
    triggers = ("记住", "以后你要知道", "我的角色是", "我日常工作是", "我主要对接", "我偏好", "我现在做的项目是")
    if not any(trigger in message for trigger in triggers):
        return []
    items: list[dict[str, str]] = []
    mappings = [
        ("role", r"我的角色是([^，。；\n]+)"),
        ("work", r"我日常工作是([^，。；\n]+)"),
        ("customer", r"我主要对接([^，。；\n]+)"),
        ("preference", r"我偏好([^，。；\n]+)"),
        ("project", r"我现在做的项目是([^，。；\n]+)"),
    ]
    for memory_type, pattern in mappings:
        match = re.search(pattern, message)
        if match:
            items.append({"memory_type": memory_type, "content": match.group(1).strip()})
    if "企业知识库 Agent" in message and not any(item["memory_type"] == "project" for item in items):
        items.append({"memory_type": "project", "content": "企业知识库 Agent"})
    if "Codex" in message and "提示词" in message and not any(item["memory_type"] == "preference" for item in items):
        items.append({"memory_type": "preference", "content": "偏好可直接发给 Codex 的提示词"})
    return items


def profile_patch_from_memory(items: list[dict[str, str]]) -> dict[str, object]:
    patch: dict[str, object] = {}
    for item in items:
        memory_type = item["memory_type"]
        content = item["content"]
        if memory_type == "role":
            patch["role"] = content
        elif memory_type == "work":
            patch["daily_work"] = content
        elif memory_type == "customer":
            patch.setdefault("customers", []).append(content)
        elif memory_type == "preference":
            patch["answer_preference"] = content
        elif memory_type == "project":
            patch.setdefault("current_projects", []).append(content)
            if "企业知识库" in content:
                patch.setdefault("business_domains", []).extend(["企业知识库", "Agent开发", "RAG"])
    return patch
