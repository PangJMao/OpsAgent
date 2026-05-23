from __future__ import annotations

import json
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ops_agent.config import settings
from ops_agent.models import utc_now_iso


@dataclass
class TraceEvent:
    node: str
    started_at: str
    duration_ms: float
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class TraceRecorder:
    """记录节点级执行细节，便于调试、复盘和后续评测。"""

    def __init__(self, trace_id: str | None = None, traces_dir: Path = settings.traces_dir) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex
        self.traces_dir = traces_dir
        self.events: list[TraceEvent] = []
        self.created_at = utc_now_iso()
        self.context: dict[str, Any] = {}

    def set_context(self, **context: Any) -> None:
        self.context.update({key: value for key, value in context.items() if value is not None})

    def record(self, node: str, input_summary: dict[str, Any] | None = None, output_summary: dict[str, Any] | None = None) -> None:
        self.events.append(
            TraceEvent(
                node=node,
                started_at=utc_now_iso(),
                duration_ms=0,
                input_summary=input_summary or {},
                output_summary=output_summary or {},
            )
        )

    def record_llm_prompt(self, node: str, messages: list[dict[str, Any]] | list[Any]) -> None:
        self.record(node, output_summary={"messages": messages})

    def record_llm_raw_output(self, node: str, raw_output: str) -> None:
        self.record(node, output_summary={"raw_output": raw_output})

    def record_parse_failure(self, node: str, raw_output: str, reason: str) -> None:
        self.record(node, output_summary={"raw_output": raw_output, "parse_failure": reason})

    def record_tool_io(self, tool: str, tool_input: dict[str, Any], observation: dict[str, Any]) -> None:
        self.record(f"tool.{tool}", input_summary={"tool_input": tool_input}, output_summary={"observation": observation})

    @contextmanager
    def span(self, node: str, input_summary: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        started_at = utc_now_iso()
        started = time.perf_counter()
        output_summary: dict[str, Any] = {}
        error: str | None = None

        try:
            yield output_summary
        except Exception as exc:
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            raise
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            self.events.append(
                TraceEvent(
                    node=node,
                    started_at=started_at,
                    duration_ms=round(duration_ms, 2),
                    input_summary=input_summary or {},
                    output_summary=output_summary,
                    error=error,
                )
            )

    def flush(self) -> Path:
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        path = self.traces_dir / f"{self.trace_id}.json"
        payload = {
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "context": self.context,
            "events": [event.__dict__ for event in self.events],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


class TraceStore:
    """Read-only access to persisted trace files."""

    def __init__(self, traces_dir: Path = settings.traces_dir) -> None:
        self.traces_dir = traces_dir

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.traces_dir.exists():
            return []

        traces = []
        for path in self.traces_dir.glob("*.json"):
            try:
                payload = _read_trace(path)
            except (OSError, json.JSONDecodeError):
                continue
            events = payload.get("events") or []
            traces.append(
                {
                    "trace_id": payload.get("trace_id", path.stem),
                    "created_at": payload.get("created_at", ""),
                    "event_count": len(events),
                    "has_error": any(bool(event.get("error")) for event in events if isinstance(event, dict)),
                    "context": payload.get("context", {}),
                }
            )

        return sorted(traces, key=lambda trace: str(trace.get("created_at", "")), reverse=True)[:limit]

    def get(self, trace_id: str) -> dict[str, Any]:
        if not trace_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Invalid trace_id.")
        path = self.traces_dir / f"{trace_id}.json"
        if not path.exists():
            raise FileNotFoundError(trace_id)
        return _read_trace(path)


def _read_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("Trace payload must be an object.", doc="", pos=0)
    return payload
