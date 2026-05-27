from __future__ import annotations

from ops_agent.context.context_schema import DecisionResult, EvidenceResult, MatchedRule
from ops_agent.services.business_rule_service import select_rules
from ops_agent.services.business_scene import parse_business_frame
from ops_agent.services.rag_workflow import RagWorkflowState


class IntentClassifier:
    def classify(self, state: RagWorkflowState) -> str:
        return state.question_type


class RuleMatcher:
    def match(self, state: RagWorkflowState) -> list[MatchedRule]:
        frame = state.business_frame or parse_business_frame(state.question, state.question_type)
        rules = select_rules(state.question_type, frame)
        matched: list[MatchedRule] = []
        for rule in rules:
            matched.append(
                MatchedRule(
                    rule_id=rule.rule_id,
                    source=rule.sources[0] if rule.sources else "",
                    topic=frame.action or state.question_type,
                    stage=frame.stage,
                    decision=rule.decision,
                    rationale=list(rule.rationale),
                    allowed_actions=list(rule.allowed_items),
                    forbidden_actions=list(rule.blocked_scripts),
                    risk_level=state.risk_level,
                    metadata={
                        "question_types": list(rule.question_types),
                        "stages": list(rule.stages),
                        "actions": list(rule.actions),
                        "subjects": list(rule.subjects),
                    },
                )
            )
        if not matched and state.sources:
            for index, source in enumerate(state.sources[:3], start=1):
                matched.append(
                    MatchedRule(
                        rule_id=f"retrieved-{index}",
                        source=source.doc_name,
                        topic=source.topic or state.question_type,
                        stage=frame.stage,
                        decision="",
                        rationale=[],
                        risk_level=state.risk_level,
                    )
                )
        return matched


class EvidenceValidator:
    def validate(self, state: RagWorkflowState, matched_rules: list[MatchedRule]) -> EvidenceResult:
        evidence = []
        sources: list[str] = []
        for rule in matched_rules:
            item = {
                "source": rule.source,
                "topic": rule.topic,
                "stage": rule.stage,
                "decision": rule.decision,
                "rationale": rule.rationale[:5],
                "allowed_actions": rule.allowed_actions[:5],
                "forbidden_actions": rule.forbidden_actions[:5],
                "risk_level": rule.risk_level,
            }
            evidence.append(item)
            if rule.source and rule.source not in sources:
                sources.append(rule.source)
        for source in state.sources:
            label = source.doc_name
            if source.sheet_name:
                label = f"{label} / {source.sheet_name}"
            if source.topic:
                label = f"{label} / {source.topic}"
            if label and label not in sources:
                sources.append(label)
        missing = []
        can_answer = state.should_answer or bool(matched_rules)
        if not can_answer:
            missing.append("未命中足够可靠的规则或知识库证据")
        return EvidenceResult(
            can_answer=can_answer,
            confidence=state.confidence,
            evidence=evidence,
            missing_evidence=missing,
            sources=sources[:8],
        )


class DecisionBuilder:
    def build(self, state: RagWorkflowState, matched_rules: list[MatchedRule], evidence: EvidenceResult) -> DecisionResult:
        if not evidence.can_answer:
            return DecisionResult(
                direct_answer="根据当前知识库暂无法确认",
                reason="缺少可验证的结构化规则或可靠证据",
                risk_level=state.risk_level,
                must_include=["暂无法确认", "缺少依据"],
                must_not_include=_default_forbidden(),
            )
        primary = matched_rules[0] if matched_rules else None
        direct = primary.decision if primary and primary.decision else _fallback_direct_answer(state)
        allowed = _dedupe([item for rule in matched_rules for item in rule.allowed_actions])
        forbidden = _dedupe([item for rule in matched_rules for item in rule.forbidden_actions])
        reason_items = _dedupe([item for rule in matched_rules for item in rule.rationale])
        reason = "；".join(reason_items[:3]) if reason_items else "已命中当前问题相关证据，结论以结构化规则和检索证据为准"
        return DecisionResult(
            direct_answer=direct,
            reason=reason,
            allowed_actions=allowed[:6],
            forbidden_actions=forbidden[:6],
            risk_notice="高风险内容需按法务/合规口径复核" if state.risk_level == "high" else "",
            risk_level=state.risk_level,
            must_include=_must_include(state, direct),
            must_not_include=_default_forbidden(),
        )


def _fallback_direct_answer(state: RagWorkflowState) -> str:
    if state.question_type == "asset_inquiry":
        stage = state.business_frame.stage if state.business_frame else ""
        if stage in {"D4-D6", "D7-D9", "D4-D9"}:
            return "不建议在该阶段做资产摸底；D10-D15 后只能在合规边界内用于还款能力判断。"
        if stage == "D10-D15":
            return "可在合规边界内谨慎了解，但只能服务于还款能力判断。"
    if state.question_type == "legal_compliance":
        return "不能把未确认的法务后果说成确定结果。"
    return "可以回答，但必须以当前命中的知识库证据为准。"


def _must_include(state: RagWorkflowState, direct: str) -> list[str]:
    items = []
    if direct:
        first = direct.split("；", 1)[0].strip("。")
        if first:
            items.append(first)
    stage = state.business_frame.stage if state.business_frame else ""
    if stage:
        items.append(stage)
    if state.question_type == "asset_inquiry":
        items.extend(["资产", "还款能力"])
        if stage in {"D4-D6", "D7-D9", "D4-D9"}:
            items.extend(["不建议", "D10-D15"])
    if state.question_type == "legal_compliance":
        items.extend(["风险提示", "法务", "复核"])
    return _dedupe(items)


def _default_forbidden() -> list[str]:
    return [
        "chunk_id",
        "score",
        "vector_score",
        "keyword_score",
        "hybrid_score",
        "rerank_score",
        "UUID",
        "Sheet3 Sheet3",
        "业务分类",
        "DO-1分",
        "字段3",
    ]


def _dedupe(items: list[str]) -> list[str]:
    return [item for item in dict.fromkeys(item for item in items if item)]
