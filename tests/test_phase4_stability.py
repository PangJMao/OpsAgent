from pathlib import Path

from fastapi.testclient import TestClient

from ops_agent.main import create_app
from ops_agent.models import Citation, RagAnswer
from ops_agent.services import EvaluationCase, EvaluationService, InMemoryTaskQueue, TraceRecorder, TraceStore
from ops_agent.services.permission_service import PermissionContext, PermissionService


def test_permission_service_blocks_user_admin_actions() -> None:
    permissions = PermissionService()
    user = PermissionContext(user_id="u1", role="user")
    admin = PermissionContext(user_id="a1", role="admin")

    assert permissions.can(user, "agent.run") is True
    assert permissions.can(user, "evaluation.run") is False
    assert permissions.can(admin, "evaluation.run") is True


def test_task_queue_records_success_and_failure() -> None:
    queue = InMemoryTaskQueue()

    succeeded = queue.submit("ok", lambda: {"value": 1})
    failed = queue.submit("fail", lambda: _raise_runtime_error())

    assert succeeded.status == "succeeded"
    assert succeeded.result == {"value": 1}
    assert failed.status == "failed"
    assert "boom" in str(failed.error)
    assert [task.name for task in queue.list()] == ["fail", "ok"]


def test_evaluation_service_calculates_metrics() -> None:
    service = EvaluationService(answer_service=FakeAnswerService())

    report = service.run(
        [
            EvaluationCase(question="高级客户售后多久响应？", expected_answer_contains="4 小时"),
            EvaluationCase(question="不存在的问题", expect_refused=True, require_citation=False),
        ]
    )

    assert report["total"] == 2
    assert report["passed"] == 2
    assert report["pass_rate"] == 1.0
    assert report["refusal_match_rate"] == 1.0


def test_phase4_routes_are_registered() -> None:
    routes = {route.path for route in create_app().routes}

    assert "/evaluation/run" in routes
    assert "/tasks" in routes
    assert "/tasks/{task_id}" in routes
    assert "/traces" in routes
    assert "/traces/{trace_id}" in routes


def test_rag_ingest_requires_admin_role() -> None:
    client = TestClient(create_app(), raise_server_exceptions=False)

    forbidden = client.post("/rag/ingest", json={"path": "missing.md"})
    allowed = client.post("/rag/ingest", json={"path": "missing.md"}, headers={"X-Role": "admin"})

    assert forbidden.status_code == 403
    assert allowed.status_code != 403


def test_trace_store_lists_and_reads_trace_context(tmp_path: Path) -> None:
    recorder = TraceRecorder(trace_id="trace_demo", traces_dir=tmp_path)
    recorder.set_context(user_id="u1", role="admin", knowledge_scopes=["default"])
    with recorder.span("demo.node", {"input": "ok"}) as span:
        span.update({"output": "ok"})
    recorder.flush()

    store = TraceStore(traces_dir=tmp_path)
    traces = store.list()
    trace = store.get("trace_demo")

    assert traces[0]["trace_id"] == "trace_demo"
    assert traces[0]["context"]["user_id"] == "u1"
    assert trace["context"]["role"] == "admin"
    assert trace["events"][0]["node"] == "demo.node"


def test_trace_routes_require_admin_role() -> None:
    client = TestClient(create_app())

    forbidden = client.get("/traces")
    allowed = client.get("/traces", headers={"X-Role": "admin"})

    assert forbidden.status_code == 403
    assert allowed.status_code == 200


class FakeAnswerService:
    def ask(self, question: str) -> RagAnswer:
        if "不存在" in question:
            return RagAnswer(
                trace_id="trace-2",
                question=question,
                answer="当前知识库没有足够依据回答该问题。",
                citations=[],
                confidence=0.0,
                refused=True,
            )
        return RagAnswer(
            trace_id="trace-1",
            question=question,
            answer="高级客户售后响应时间为 4 小时内。",
            citations=[
                Citation(
                    document_id="doc",
                    title="售后政策",
                    chunk_id="chunk",
                    score=1.0,
                )
            ],
            confidence=1.0,
            refused=False,
        )


def _raise_runtime_error() -> dict[str, object]:
    raise RuntimeError("boom")
