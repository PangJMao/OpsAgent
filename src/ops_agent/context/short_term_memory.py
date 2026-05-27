from __future__ import annotations

from copy import deepcopy

from ops_agent.context.context_compressor import ContextCompressor
from ops_agent.context.context_schema import ConversationMessage, ConversationState, ShortTermMemory
from ops_agent.context.conversation_state import update_state_from_message


class ShortTermMemoryStore:
    def __init__(self, compressor: ContextCompressor | None = None) -> None:
        self._sessions: dict[str, ShortTermMemory] = {}
        self._states: dict[str, ConversationState] = {}
        self.compressor = compressor or ContextCompressor()

    def load(self, session_id: str) -> ShortTermMemory:
        return deepcopy(self._sessions.get(session_id, ShortTermMemory()))

    def load_state(self, session_id: str) -> ConversationState:
        return deepcopy(self._states.get(session_id, ConversationState()))

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> ShortTermMemory:
        memory = self._sessions.setdefault(session_id, ShortTermMemory())
        memory.recent_messages.append(ConversationMessage(role=role, content=content, metadata=metadata or {}))
        if self.compressor.should_compress(memory.recent_messages):
            memory.conversation_summary = self.compressor.compress_conversation(
                memory.recent_messages,
                memory.conversation_summary,
            )
            memory.recent_messages = memory.recent_messages[-self.compressor.keep_recent :]
        return deepcopy(memory)

    def update_state(
        self,
        session_id: str,
        message: str,
        *,
        intent: str = "",
        decision: str = "",
        sources: list[str] | None = None,
    ) -> ConversationState:
        current = self._states.get(session_id, ConversationState())
        updated = update_state_from_message(current, message, intent=intent, decision=decision, sources=sources)
        self._states[session_id] = updated
        return deepcopy(updated)

    def save_state(self, session_id: str, state: ConversationState) -> None:
        self._states[session_id] = deepcopy(state)


short_term_memory_store = ShortTermMemoryStore()
