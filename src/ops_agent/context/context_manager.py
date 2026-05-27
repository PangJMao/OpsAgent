from __future__ import annotations

from dataclasses import asdict

from ops_agent.context.context_assembler import ContextAssembler
from ops_agent.context.context_compressor import ContextCompressor
from ops_agent.context.context_policy import ContextPolicy
from ops_agent.context.context_schema import AgentContext, DecisionResult, EvidenceResult, LLMContext, RiskContext
from ops_agent.context.conversation_state import resolve_coreference, update_state_from_message
from ops_agent.context.long_term_memory import LongTermMemoryRepository
from ops_agent.context.memory_updater import MemoryUpdater, extract_explicit_memory
from ops_agent.context.short_term_memory import ShortTermMemoryStore, short_term_memory_store
from ops_agent.context.user_profile import UserProfileRepository


DEFAULT_USER_ID = "default_user"
DEFAULT_SESSION_ID = "default_session"


class ContextManager:
    def __init__(
        self,
        short_term: ShortTermMemoryStore | None = None,
        long_term: LongTermMemoryRepository | None = None,
        profiles: UserProfileRepository | None = None,
        compressor: ContextCompressor | None = None,
        policy: ContextPolicy | None = None,
    ) -> None:
        self.short_term = short_term or short_term_memory_store
        self.long_term = long_term or LongTermMemoryRepository()
        self.profiles = profiles or UserProfileRepository()
        self.compressor = compressor or ContextCompressor()
        self.policy = policy or ContextPolicy()
        self.assembler = ContextAssembler(self.policy)
        self.updater = MemoryUpdater(self.short_term, self.long_term, self.profiles)

    def build_context(self, user_id: str, session_id: str, user_message: str) -> AgentContext:
        user_id = user_id or DEFAULT_USER_ID
        session_id = session_id or DEFAULT_SESSION_ID
        conversation_state = self.short_term.load_state(session_id)
        short_memory = self.short_term.load(session_id)
        resolved = resolve_coreference(user_message, conversation_state)
        updated_state = update_state_from_message(conversation_state, resolved)
        long_memory = self.long_term.load(user_id)
        profile = self.profiles.load_user_profile(user_id)
        risk = RiskContext(risk_level="low", risk_reasons=[], requires_escalation=False)
        candidate_memory = extract_explicit_memory(user_message)
        return AgentContext(
            user_id=user_id,
            session_id=session_id,
            current_message=user_message,
            resolved_message=resolved,
            conversation_state=updated_state,
            short_term_memory=short_memory,
            long_term_memory=long_memory,
            user_profile=profile,
            risk_context=risk,
            candidate_memory=candidate_memory,
        )

    def assemble_llm_context(
        self,
        agent_context: AgentContext,
        decision: DecisionResult,
        evidence: EvidenceResult,
        *,
        intent: str,
    ) -> LLMContext:
        return self.assembler.assemble(agent_context, decision, evidence, intent=intent)

    def update_after_response(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        answer: str,
        metadata: dict,
    ) -> None:
        context = self.build_context(user_id, session_id, user_message)
        self.updater.update_after_response(context, answer, metadata)

    def snapshot(self, context: AgentContext) -> dict[str, object]:
        payload = asdict(context)
        if context.user_profile is None:
            payload["user_profile"] = None
        return payload
