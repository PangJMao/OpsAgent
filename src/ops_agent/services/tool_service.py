from __future__ import annotations

from ops_agent.models import ToolCall, ToolResult


class ToolRegistry:
    """Whitelist for business tools that the agent is allowed to execute."""

    def __init__(self) -> None:
        self._tools = {
            "search_customer": self._search_customer,
            "draft_followup_email": self._draft_followup_email,
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
