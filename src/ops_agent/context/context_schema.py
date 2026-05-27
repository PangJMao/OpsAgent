from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ops_agent.models import utc_now_iso


@dataclass
class ConversationMessage:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class ConversationSummary:
    summary: str = ""
    current_project: str = ""
    current_goal: str = ""
    key_decisions: list[str] = field(default_factory=list)
    open_issues: list[str] = field(default_factory=list)
    last_updated_at: str = field(default_factory=utc_now_iso)


@dataclass
class ConversationState:
    current_topic: str = ""
    current_stage: str = ""
    current_scene: str = ""
    last_intent: str = ""
    last_decision: str = ""
    last_sources: list[str] = field(default_factory=list)


@dataclass
class ShortTermMemory:
    recent_messages: list[ConversationMessage] = field(default_factory=list)
    conversation_summary: ConversationSummary = field(default_factory=ConversationSummary)
    active_constraints: list[str] = field(default_factory=list)
    mentioned_entities: list[str] = field(default_factory=list)
    pending_questions: list[str] = field(default_factory=list)


@dataclass
class LongTermMemoryItem:
    memory_id: str
    user_id: str
    memory_type: str
    content: str
    confidence: float = 1.0
    source: str = "explicit"
    source_message: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    is_active: bool = True


@dataclass
class LongTermMemory:
    stable_user_facts: list[str] = field(default_factory=list)
    work_preferences: list[str] = field(default_factory=list)
    frequent_business_scenes: list[str] = field(default_factory=list)
    saved_constraints: list[str] = field(default_factory=list)
    items: list[LongTermMemoryItem] = field(default_factory=list)


@dataclass
class UserProfile:
    user_id: str
    role: str = ""
    daily_work: str = ""
    business_domains: list[str] = field(default_factory=list)
    customers: list[str] = field(default_factory=list)
    current_projects: list[str] = field(default_factory=list)
    skill_level: str = ""
    answer_preference: str = ""
    preferences: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass
class MatchedRule:
    rule_id: str
    source: str
    topic: str
    stage: str = ""
    decision: str = ""
    rationale: list[str] = field(default_factory=list)
    allowed_actions: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    risk_level: str = "low"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceResult:
    can_answer: bool = False
    confidence: float = 0.0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


@dataclass
class DecisionResult:
    direct_answer: str = ""
    reason: str = ""
    allowed_actions: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    risk_notice: str = ""
    risk_level: str = "low"
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)


@dataclass
class RetrievalContext:
    rewritten_queries: list[str] = field(default_factory=list)
    matched_rules: list[MatchedRule] = field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RiskContext:
    risk_level: str = "low"
    risk_reasons: list[str] = field(default_factory=list)
    requires_escalation: bool = False


@dataclass
class AgentContext:
    user_id: str
    session_id: str
    current_message: str
    resolved_message: str = ""
    conversation_state: ConversationState = field(default_factory=ConversationState)
    short_term_memory: ShortTermMemory = field(default_factory=ShortTermMemory)
    long_term_memory: LongTermMemory = field(default_factory=LongTermMemory)
    user_profile: UserProfile | None = None
    retrieval_context: RetrievalContext = field(default_factory=RetrievalContext)
    risk_context: RiskContext = field(default_factory=RiskContext)
    candidate_memory: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LLMContext:
    user_question: str
    intent: str
    conversation_state: dict[str, Any]
    decision: dict[str, Any]
    evidence: list[dict[str, Any]]
    memory_context: dict[str, Any]
    must_include: list[str]
    must_not_include: list[str]
    risk_level: str
    forbidden_context: list[str]
    citations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
