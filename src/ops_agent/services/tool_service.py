from __future__ import annotations

import hashlib

from ops_agent.models import ToolCall, ToolResult


class ToolRegistry:
    """Whitelist for business tools that the agent is allowed to execute."""

    def __init__(self) -> None:
        self._tools = {
            "search_customer": self._search_customer,
            "draft_followup_email": self._draft_followup_email,
            "create_ticket": self._create_ticket,
            "summarize_customer_visit": self._summarize_customer_visit,
        }

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.tool)
        if tool is None:
            return ToolResult(tool=call.tool, ok=False, error="Tool is not whitelisted.")
        return tool(call.args)

    def _search_customer(self, args: dict[str, object]) -> ToolResult:
        company_name = str(args.get("company_name") or "未知客户").strip()
        return ToolResult(
            tool="search_customer",
            ok=True,
            result={
                "company_name": company_name,
                "tier": "enterprise",
                "owner": "sales_ops_demo",
                "last_contact": "2026-05-20",
            },
        )

    def _draft_followup_email(self, args: dict[str, object]) -> ToolResult:
        company_name = str(args.get("company_name") or "客户").strip()
        topic = str(args.get("topic") or "后续沟通").strip()
        return ToolResult(
            tool="draft_followup_email",
            ok=True,
            result={
                "subject": f"{company_name} - {topic}",
                "body": f"您好，关于{topic}，我们已整理出初步信息，建议安排一次后续沟通确认细节。",
            },
        )

    def _create_ticket(self, args: dict[str, object]) -> ToolResult:
        company_name = str(args.get("company_name") or "客户").strip()
        title = str(args.get("title") or "客户问题跟进").strip()
        priority = str(args.get("priority") or "medium").strip().lower()
        if priority not in {"low", "medium", "high"}:
            priority = "medium"
        return ToolResult(
            tool="create_ticket",
            ok=True,
            result={
                "ticket_id": f"TICKET-{_stable_id(company_name, title)}",
                "company_name": company_name,
                "title": title,
                "priority": priority,
                "status": "pending_human_confirm",
                "assignee": "support_ops_demo",
            },
        )

    def _summarize_customer_visit(self, args: dict[str, object]) -> ToolResult:
        company_name = str(args.get("company_name") or "客户").strip()
        notes = str(args.get("notes") or "客户拜访记录待补充").strip()
        return ToolResult(
            tool="summarize_customer_visit",
            ok=True,
            result={
                "company_name": company_name,
                "summary": f"{company_name}本次沟通重点：{notes}",
                "customer_concerns": ["预算范围", "上线周期", "售后响应"],
                "next_steps": ["确认决策人", "发送跟进邮件", "安排方案复盘"],
            },
        )


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return f"{int(digest[:8], 16) % 100000:05d}"
