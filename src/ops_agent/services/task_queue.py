from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal
import uuid

from ops_agent.models import utc_now_iso


TaskStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class TaskRecord:
    task_id: str
    name: str
    status: TaskStatus
    created_at: str
    updated_at: str
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class InMemoryTaskQueue:
    """Deterministic local task queue used before introducing Redis/RQ."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}

    def submit(self, name: str, handler: Callable[[], dict[str, Any]]) -> TaskRecord:
        now = utc_now_iso()
        record = TaskRecord(
            task_id=uuid.uuid4().hex,
            name=name,
            status="queued",
            created_at=now,
            updated_at=now,
        )
        self._tasks[record.task_id] = record
        return self.run(record.task_id, handler)

    def run(self, task_id: str, handler: Callable[[], dict[str, Any]]) -> TaskRecord:
        record = self.get(task_id)
        record.status = "running"
        record.updated_at = utc_now_iso()
        try:
            record.result = handler()
            record.status = "succeeded"
            record.error = None
        except Exception as exc:
            record.result = {}
            record.status = "failed"
            record.error = str(exc)
        record.updated_at = utc_now_iso()
        return record

    def get(self, task_id: str) -> TaskRecord:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"Task not found: {task_id}") from exc

    def list(self) -> list[TaskRecord]:
        return sorted(self._tasks.values(), key=lambda task: task.created_at, reverse=True)


task_queue = InMemoryTaskQueue()
