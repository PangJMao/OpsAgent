from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Role = Literal["user", "admin", "root"]
Action = Literal["rag.ask", "rag.ingest", "agent.run", "trace.read", "evaluation.run", "task.read"]


@dataclass(frozen=True)
class PermissionContext:
    user_id: str
    role: Role = "user"
    knowledge_scopes: tuple[str, ...] = ("default",)


class PermissionService:
    """Small role-based permission gate for phase-4 API hardening."""

    _role_actions: dict[Role, set[Action]] = {
        "user": {"rag.ask", "agent.run", "task.read"},
        "admin": {"rag.ask", "rag.ingest", "agent.run", "trace.read", "evaluation.run", "task.read"},
        "root": {"rag.ask", "rag.ingest", "agent.run", "trace.read", "evaluation.run", "task.read"},
    }

    def can(self, context: PermissionContext, action: Action, scope: str = "default") -> bool:
        allowed_actions = self._role_actions.get(context.role, set())
        if action not in allowed_actions:
            return False
        return context.role in {"admin", "root"} or scope in context.knowledge_scopes

    def require(self, context: PermissionContext, action: Action, scope: str = "default") -> None:
        if not self.can(context, action, scope):
            raise PermissionError(f"{context.role} cannot perform {action} on scope {scope}.")


def context_from_headers(
    user_id: str | None,
    role: str | None,
    scopes: str | None,
) -> PermissionContext:
    normalized_role: Role = "root" if role == "root" else "admin" if role == "admin" else "user"
    normalized_scopes = tuple(scope.strip() for scope in (scopes or "default").split(",") if scope.strip())
    return PermissionContext(
        user_id=user_id or "anonymous",
        role=normalized_role,
        knowledge_scopes=normalized_scopes or ("default",),
    )
