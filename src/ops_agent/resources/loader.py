from __future__ import annotations

from pathlib import Path


class ResourceLoader:
    """Loads small business resources that can be injected into prompts."""

    def __init__(self, resource_dir: Path | None = None, max_chars: int = 2000) -> None:
        self.resource_dir = resource_dir or Path(__file__).resolve().parent
        self.max_chars = max_chars

    def load(self, resource_name: str) -> str:
        path = self.resource_dir / resource_name
        if not path.exists():
            raise FileNotFoundError(f"Prompt resource not found: {resource_name}")

        content = path.read_text(encoding="utf-8").strip()
        # 资源注入设置长度上限，避免业务资料越堆越多导致上下文不可控。
        if len(content) <= self.max_chars:
            return content
        return f"{content[: self.max_chars]}\n\n[resource truncated]"
