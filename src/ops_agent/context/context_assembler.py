from __future__ import annotations

from dataclasses import asdict

from ops_agent.context.context_policy import ContextPolicy
from ops_agent.context.context_schema import AgentContext, DecisionResult, EvidenceResult, LLMContext


class ContextAssembler:
    def __init__(self, policy: ContextPolicy | None = None) -> None:
        self.policy = policy or ContextPolicy()

    def assemble(
        self,
        agent_context: AgentContext,
        decision: DecisionResult,
        evidence: EvidenceResult,
        *,
        intent: str,
    ) -> LLMContext:
        filtered_evidence = self.policy.filter_evidence(evidence, intent)
        filtered_memories = self.policy.filter_memory(agent_context.long_term_memory.items, intent)
        profile = self.policy.filter_profile(agent_context.user_profile, intent)
        summary = agent_context.short_term_memory.conversation_summary
        memory_context = {
            "short_term": {
                "conversation_summary": asdict(summary),
                "recent_messages": [
                    {"role": message.role, "content": message.content}
                    for message in agent_context.short_term_memory.recent_messages[-5:]
                ],
                "active_constraints": agent_context.short_term_memory.active_constraints,
            },
            "long_term": [
                {"type": memory.memory_type, "content": memory.content}
                for memory in filtered_memories
            ],
            "profile": profile,
        }
        return LLMContext(
            user_question=agent_context.resolved_message or agent_context.current_message,
            intent=intent,
            conversation_state=asdict(agent_context.conversation_state),
            decision=asdict(decision),
            evidence=filtered_evidence.evidence,
            memory_context=memory_context,
            must_include=decision.must_include,
            must_not_include=decision.must_not_include,
            risk_level=decision.risk_level or agent_context.risk_context.risk_level,
            forbidden_context=self.policy.build_forbidden_context(intent),
            citations=filtered_evidence.sources,
        )
