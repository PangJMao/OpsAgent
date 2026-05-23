from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeState:
    status: str = "starting"
    startup_errors: list[str] = field(default_factory=list)
    components: dict[str, dict[str, Any]] = field(default_factory=dict)

    def mark_ready(self, component: str, detail: dict[str, Any] | None = None) -> None:
        self.components[component] = {"status": "ok", **(detail or {})}
        self.status = "degraded" if self.startup_errors else "ok"

    def mark_degraded(self, component: str, message: str, detail: dict[str, Any] | None = None) -> None:
        self.components[component] = {"status": "degraded", "message": message, **(detail or {})}
        self.startup_errors.append(f"{component}: {message}")
        self.status = "degraded"

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "startup_errors": list(self.startup_errors),
            "components": self.components,
        }


runtime_state = RuntimeState()
