from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptRenderInput:
    question: str
    route: str
    evidence: str
    tool_results: str
    business_resources: str


class PromptManager:
    """Centralized prompt template reader and renderer."""

    def __init__(self, template_dir: Path | None = None) -> None:
        self.template_dir = template_dir or Path(__file__).resolve().parent / "templates"

    def render(self, template_name: str, data: PromptRenderInput) -> str:
        template = self._read_template(template_name)
        values = {
            "question": data.question,
            "route": data.route,
            "evidence": data.evidence,
            "tool_results": data.tool_results,
            "business_resources": data.business_resources,
        }
        return template.format_map(values)

    def _read_template(self, template_name: str) -> str:
        path = self.template_dir / template_name
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_name}")
        return path.read_text(encoding="utf-8")
