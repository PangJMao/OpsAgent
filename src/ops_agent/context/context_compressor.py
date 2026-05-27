from __future__ import annotations

from ops_agent.context.context_schema import ConversationMessage, ConversationSummary, MatchedRule
from ops_agent.models import utc_now_iso


class ContextCompressor:
    def __init__(self, keep_recent: int = 6, max_messages: int = 10, max_chars: int = 6000) -> None:
        self.keep_recent = keep_recent
        self.max_messages = max_messages
        self.max_chars = max_chars

    def should_compress(self, messages: list[ConversationMessage]) -> bool:
        total_chars = sum(len(message.content) for message in messages)
        return len(messages) > self.max_messages or total_chars > self.max_chars

    def compress_conversation(
        self,
        messages: list[ConversationMessage],
        existing: ConversationSummary | None = None,
    ) -> ConversationSummary:
        older = messages[:-self.keep_recent] if len(messages) > self.keep_recent else messages
        existing = existing or ConversationSummary()
        facts = _extract_summary_facts(older)
        key_decisions = list(dict.fromkeys([*existing.key_decisions, *facts["key_decisions"]]))[-8:]
        open_issues = list(dict.fromkeys([*existing.open_issues, *facts["open_issues"]]))[-8:]
        summary_parts = [existing.summary] if existing.summary else []
        if facts["summary"]:
            summary_parts.append(facts["summary"])
        return ConversationSummary(
            summary="；".join(part for part in summary_parts if part)[-1200:],
            current_project=facts["current_project"] or existing.current_project,
            current_goal=facts["current_goal"] or existing.current_goal,
            key_decisions=key_decisions,
            open_issues=open_issues,
            last_updated_at=utc_now_iso(),
        )

    def compress_rules(self, matched_rules: list[MatchedRule], limit: int = 5) -> list[MatchedRule]:
        result: list[MatchedRule] = []
        seen: set[str] = set()
        for rule in matched_rules:
            key = rule.rule_id or f"{rule.source}:{rule.topic}:{rule.stage}:{rule.decision}"
            if key in seen:
                continue
            seen.add(key)
            result.append(rule)
            if len(result) >= limit:
                break
        return result


def _extract_summary_facts(messages: list[ConversationMessage]) -> dict[str, object]:
    lines = [message.content.strip() for message in messages if message.content.strip()]
    text = "\n".join(lines)
    key_decisions: list[str] = []
    open_issues: list[str] = []
    if "结构化规则" in text or "Rule RAG" in text:
        key_decisions.append("优先实现结构化规则型 RAG 和上下文工程")
    if "短期记忆" in text:
        key_decisions.append("短期记忆用于当前会话状态和多轮追问补全")
    if "长期记忆" in text or "用户画像" in text:
        key_decisions.append("长期记忆和用户画像仅保存稳定用户信息")
    if "规则匹配" in text:
        open_issues.append("规则匹配和证据校验需要持续接入企业知识库")
    if "评测" in text or "evaluation" in text:
        open_issues.append("需要通过 evaluation harness 验证回答质量")
    current_project = "企业知识库 Agent" if "企业知识库" in text or "Agent" in text else ""
    current_goal = "搭建结构化规则型 RAG 和上下文工程" if "上下文工程" in text or "Structured Rule RAG" in text else ""
    compact = " / ".join(lines[-6:])
    return {
        "summary": compact[-800:],
        "current_project": current_project,
        "current_goal": current_goal,
        "key_decisions": key_decisions,
        "open_issues": open_issues,
    }
