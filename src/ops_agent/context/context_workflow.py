from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ContextWorkflowSpec:
    """LangGraph-compatible node order for the context engineering path.

    The project does not require langgraph at import time; this spec keeps the
    workflow explicit and can be used to build a StateGraph when the dependency
    is enabled.
    """

    nodes: list[str] = field(
        default_factory=lambda: [
            "load_context",
            "resolve_coreference",
            "classify_intent",
            "route_by_intent",
            "retrieve_or_match_rules",
            "validate_evidence",
            "build_decision",
            "compress_context",
            "assemble_llm_context",
            "generate_answer",
            "format_citations",
            "update_memory",
        ]
    )

    def as_edges(self) -> list[tuple[str, str]]:
        return list(zip(["START", *self.nodes], [*self.nodes, "END"], strict=True))


class ContextEngineeringWorkflow:
    """Small deterministic runner mirroring the intended LangGraph nodes."""

    def __init__(self, handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]]) -> None:
        self.spec = ContextWorkflowSpec()
        self.handlers = handlers

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        for node in self.spec.nodes:
            handler = self.handlers.get(node)
            if handler is None:
                continue
            state = handler(state)
            state.setdefault("visited_nodes", []).append(node)
        return state
