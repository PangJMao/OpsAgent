from __future__ import annotations

from ops_agent.context.context_schema import EvidenceResult, LongTermMemoryItem, MatchedRule, UserProfile


LEGAL_INTENTS = {"legal_compliance", "legal_script", "false_commitment"}
BUSINESS_RULE_INTENTS = {
    "asset_inquiry",
    "funding_check",
    "contact_boundary",
    "stage_script",
    "legal_compliance",
    "false_commitment",
}


class ContextPolicy:
    def filter_memory(self, memories: list[LongTermMemoryItem], intent: str) -> list[LongTermMemoryItem]:
        allowed_types = {"role", "work", "project", "preference", "constraint"}
        if intent in BUSINESS_RULE_INTENTS:
            allowed_types = {"role", "work", "project", "preference"}
        return [
            memory
            for memory in memories
            if memory.is_active and memory.memory_type in allowed_types and not _contains_sensitive_business_fact(memory.content)
        ][:6]

    def filter_profile(self, profile: UserProfile | None, intent: str) -> dict[str, object]:
        if profile is None:
            return {}
        payload: dict[str, object] = {
            "role": profile.role,
            "daily_work": profile.daily_work,
            "business_domains": profile.business_domains[:5],
            "current_projects": profile.current_projects[:5],
            "skill_level": profile.skill_level,
            "answer_preference": profile.answer_preference,
        }
        if intent in BUSINESS_RULE_INTENTS:
            return {key: value for key, value in payload.items() if key in {"role", "daily_work", "answer_preference", "current_projects"} and value}
        return {key: value for key, value in payload.items() if value}

    def filter_evidence(self, evidence: EvidenceResult, intent: str) -> EvidenceResult:
        if intent in LEGAL_INTENTS:
            return evidence
        filtered = []
        for item in evidence.evidence:
            source = str(item.get("source") or item.get("doc_name") or "")
            topic = str(item.get("topic") or "")
            if "法务" in source or "诉讼" in topic:
                continue
            filtered.append(item)
        return EvidenceResult(
            can_answer=evidence.can_answer,
            confidence=evidence.confidence,
            evidence=filtered,
            missing_evidence=evidence.missing_evidence,
            sources=[source for source in evidence.sources if intent in LEGAL_INTENTS or "法务" not in source],
        )

    def filter_rules(self, rules: list[MatchedRule], intent: str) -> list[MatchedRule]:
        if intent in LEGAL_INTENTS:
            return rules
        return [rule for rule in rules if "法务" not in rule.source and rule.topic != "法务"] or rules[:1]

    def build_forbidden_context(self, intent: str) -> list[str]:
        forbidden = [
            "chunk_id",
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
        if intent not in LEGAL_INTENTS:
            forbidden.append("普通沟通问题不得默认注入《法务话术》作为主要依据")
        forbidden.append("用户画像不能覆盖知识库规则或证据校验结论")
        return forbidden


def _contains_sensitive_business_fact(content: str) -> bool:
    return any(term in content for term in ("D4-D6 不建议", "D7-D9 不建议", "冻结银行卡", "资产摸底规则"))
