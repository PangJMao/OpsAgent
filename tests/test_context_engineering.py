from pathlib import Path

from ops_agent.context import ContextManager, ContextPolicy
from ops_agent.context.context_compressor import ContextCompressor
from ops_agent.context.context_schema import ConversationMessage, EvidenceResult
from ops_agent.context.long_term_memory import LongTermMemoryRepository
from ops_agent.context.short_term_memory import ShortTermMemoryStore
from ops_agent.context.user_profile import UserProfileRepository
from ops_agent.services.rag_workflow import RagWorkflowState
from ops_agent.services.structured_rule_rag import DecisionBuilder, EvidenceValidator, RuleMatcher


def _manager(tmp_path: Path) -> ContextManager:
    short_term = ShortTermMemoryStore(ContextCompressor(keep_recent=4, max_messages=10, max_chars=4000))
    return ContextManager(
        short_term=short_term,
        long_term=LongTermMemoryRepository(tmp_path / "long_term.json"),
        profiles=UserProfileRepository(tmp_path / "profiles.json"),
    )


def test_context_resolve_stage_followup(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.update_after_response(
        "u1",
        "s1",
        "D4-D6 阶段客户应该怎么沟通？",
        "D4-D6 属于早期阶段，应中性温和，以信息核实为主。",
        {"intent": "stage_script", "decision": "D4-D6 中性温和", "sources": ["核资话术分层建议"]},
    )

    context = manager.build_context("u1", "s1", "那能问资产吗？")
    state = context.conversation_state

    assert context.resolved_message == "D4-D6 阶段：能问资产吗？"
    assert state.current_stage == "D4-D6"
    assert state.current_topic == "资产摸底"


def test_short_term_memory_conversation_state(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.update_after_response(
        "u1",
        "s1",
        "D4-D6 阶段客户应该怎么沟通？",
        "D4-D6 阶段以温和提醒为主。",
        {"intent": "stage_script", "decision": "温和提醒", "sources": []},
    )
    manager.update_after_response(
        "u1",
        "s1",
        "那能问资产吗？",
        "D4-D6 不建议资产摸底。",
        {"intent": "asset_inquiry", "decision": "不建议", "sources": []},
    )

    state = manager.short_term.load_state("s1")

    assert state.current_stage == "D4-D6"
    assert state.current_topic == "资产摸底"
    assert state.last_intent == "asset_inquiry"


def test_long_term_memory_explicit_save(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    manager.update_after_response(
        "u1",
        "s1",
        "记住，我现在主要在做企业知识库 Agent 项目，偏好你给我可直接发给 Codex 的提示词。",
        "已记录。",
        {"intent": "memory_update", "decision": "saved", "sources": []},
    )

    memory = manager.long_term.load("u1")
    profile = manager.profiles.load_user_profile("u1")

    assert any("企业知识库 Agent" in item.content for item in memory.items)
    assert any("Codex" in item.content for item in memory.items)
    assert "企业知识库 Agent" in profile.current_projects
    assert "Codex" in profile.answer_preference


def test_profile_load_into_context(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.profiles.update_user_profile(
        "u1",
        {
            "role": "企业知识库开发者",
            "answer_preference": "架构分析和代码方案",
            "current_projects": ["企业知识库 Agent"],
        },
    )

    context = manager.build_context("u1", "s1", "D4-D6 阶段能问资产吗？")
    llm_context = manager.assemble_llm_context(
        context,
        DecisionBuilder().build(
            RagWorkflowState(question=context.resolved_message, question_type="asset_inquiry", risk_level="high"),
            [],
            EvidenceResult(can_answer=True, confidence=0.8, evidence=[{"source": "核资话术分层建议"}], sources=["核资话术分层建议"]),
        ),
        EvidenceResult(can_answer=True, confidence=0.8, evidence=[{"source": "核资话术分层建议"}], sources=["核资话术分层建议"]),
        intent="asset_inquiry",
    )

    assert llm_context.memory_context["profile"]["role"] == "企业知识库开发者"
    assert llm_context.evidence[0]["source"] == "核资话术分层建议"


def test_context_policy_legal_filter() -> None:
    policy = ContextPolicy()
    evidence = EvidenceResult(
        can_answer=True,
        confidence=0.9,
        evidence=[
            {"source": "话术-沟通话术四大框架", "topic": "客户沟通"},
            {"source": "法务话术", "topic": "诉讼表达"},
        ],
        sources=["话术-沟通话术四大框架", "法务话术"],
    )

    normal = policy.filter_evidence(evidence, "communication_script")
    legal = policy.filter_evidence(evidence, "legal_compliance")

    assert all("法务" not in item["source"] for item in normal.evidence)
    assert any("法务" in item["source"] for item in legal.evidence)


def test_context_compression(tmp_path: Path) -> None:
    store = ShortTermMemoryStore(ContextCompressor(keep_recent=4, max_messages=10, max_chars=10000))
    for index in range(12):
        store.append_message("s1", "user", f"第 {index} 轮：企业知识库 Agent 上下文工程和结构化规则讨论")
    store.update_state("s1", "D4-D6 阶段资产摸底怎么处理？", intent="asset_inquiry", decision="不建议")

    memory = store.load("s1")
    state = store.load_state("s1")

    assert memory.conversation_summary.summary
    assert 3 <= len(memory.recent_messages) <= 5
    assert state.current_stage == "D4-D6"
    assert state.current_topic == "资产摸底"
    assert state.last_decision == "不建议"


def test_memory_not_save_business_rules(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    manager.update_after_response(
        "u1",
        "s1",
        "D4-D6 阶段能不能问客户资产？",
        "D4-D6 不建议资产摸底，D10-D15 后可谨慎使用。",
        {"intent": "asset_inquiry", "decision": "不建议", "sources": ["核资话术分层建议"]},
    )

    memory = manager.long_term.load("u1")

    assert memory.items == []


def test_rule_rag_interfaces_build_structured_outputs() -> None:
    state = RagWorkflowState(question="D4-D6 阶段能不能问客户资产？", question_type="asset_inquiry", risk_level="high")
    state.should_answer = True
    state.confidence = 0.8
    state.business_frame = None

    rules = RuleMatcher().match(state)
    evidence = EvidenceValidator().validate(state, rules)
    decision = DecisionBuilder().build(state, rules, evidence)

    assert evidence.can_answer is True
    assert decision.direct_answer
    assert "chunk_id" in decision.must_not_include
