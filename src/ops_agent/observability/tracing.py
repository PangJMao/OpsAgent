from __future__ import annotations

import json
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ops_agent.core.config import settings
from ops_agent.schemas.rag import utc_now_iso


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
            "events": [event.__dict__ for event in self.events],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
