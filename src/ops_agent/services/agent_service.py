from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Callable, Protocol, TypeVar

from ops_agent.config import settings
from ops_agent.models import AgentAnswer, AgentReview, Citation, RetrievalHit, ToolCall, ToolResult
from ops_agent.models.domain import RouteType
from ops_agent.prompts import PromptManager, PromptRenderInput
from ops_agent.resources import ResourceLoader
from ops_agent.services.llm_client import DeepSeekChatClient, LlmMessage
from ops_agent.services.tool_service import ToolRegistry
from ops_agent.services.trace_service import TraceRecorder
from ops_agent.services.vector_store import LocalVectorStore

T = TypeVar("T")


class ChatClient(Protocol):
    enabled: bool

    def complete(self, messages: list[LlmMessage], temperature: float = 0.2) -> str:
        ...


@dataclass(frozen=True)
class AgentRunCommand:
    question: str


class AgentService:
    """Service layer for the complete Agent business workflow."""

    def __init__(
        self,
        vector_store: LocalVectorStore | None = None,
        recorder: TraceRecorder | None = None,
        llm: ChatClient | None = None,
        tools: ToolRegistry | None = None,
        prompt_manager: PromptManager | None = None,
        resource_loader: ResourceLoader | None = None,
    ) -> None:
        self.vector_store = vector_store or LocalVectorStore()
        self.recorder = recorder or TraceRecorder()
        self.llm = llm or DeepSeekChatClient()
        self.tools = tools or ToolRegistry()
        self.prompt_manager = prompt_manager or PromptManager()
        self.resource_loader = resource_loader or ResourceLoader()

    def run(self, command: AgentRunCommand | str) -> AgentAnswer:
        question = command if isinstance(command, str) else command.question
        question = question.strip()
        with self.recorder.span("agent.input.validate", {"question": _compact(question)}) as span:
            if not question:
                raise ValueError("问题不能为空。")
            span.update({"characters": len(question)})

        route = self._run_node("agent.router", lambda: self._route(question))
        hits = self._run_node("agent.retriever", lambda: self._retrieve(question))
        citations = [_citation_from_hit(hit) for hit in hits if hit.score > 0]
        best_score = hits[0].score if hits else 0.0

        tool_results: list[ToolResult] = []
        if route in {"tool_call", "hybrid"}:
            tool_calls = self._run_node("agent.tool.plan", lambda: self._plan_tools(question))
            tool_results = self._run_node("agent.tool.execute", lambda: self._execute_tools(tool_calls))

        # 检索置信度是 Agent 的第一道安全闸：纯知识问答没有证据时直接拒答。
        if route == "knowledge_qa" and best_score < settings.min_relevance_score:
            answer = self._refusal(question, route, best_score, tool_results)
            self._record_answer(answer)
            return answer

        draft = self._run_node(
            "agent.reasoner",
            lambda: self._generate_answer(question, route, hits, citations, tool_results),
        )
        review = self._run_node("agent.review", lambda: self._review(draft, citations, tool_results))

        # 审核失败时不返回未经校验的模型输出，统一进入保守降级。
        if not review.passed:
            answer = AgentAnswer(
                trace_id=self.recorder.trace_id,
                question=question,
                route=route,
                answer=review.final_answer or "当前结果未通过审核，请补充依据或稍后重试。",
                citations=citations,
                confidence=round(best_score, 4),
                refused=True,
                tool_results=tool_results,
                review=review,
            )
            self._record_answer(answer)
            return answer

        answer = AgentAnswer(
            trace_id=self.recorder.trace_id,
            question=question,
            route=route,
            answer=review.final_answer,
            citations=citations,
            confidence=round(best_score, 4),
            refused=False,
            tool_results=tool_results,
            review=review,
        )
        self._record_answer(answer)
        return answer

    def _run_node(self, node: str, action: Callable[[], T]) -> T:
        last_error: Exception | None = None
        for attempt in range(1, settings.agent_max_retries + 2):
            with self.recorder.span(node, {"attempt": attempt}) as span:
                try:
                    result = action()
                except Exception as exc:
                    span.update({"ok": False, "error": str(exc)})
                    last_error = exc
                    continue
                span.update({"ok": True, "summary": _summarize(result)})
                return result
        raise RuntimeError(f"{node} failed after retries: {last_error}")

    def _route(self, question: str) -> RouteType:
        lowered = question.lower()
        knowledge_keywords = ("政策", "规则", "制度", "知识库", "售后", "响应")
        wants_tool = (
            any(keyword in lowered for keyword in ("crm", "邮件", "跟进"))
            or ("查询" in lowered and "客户" in lowered)
        )
        wants_knowledge = any(keyword in lowered for keyword in knowledge_keywords)
        if wants_tool and wants_knowledge:
            return "hybrid"
        if wants_tool:
            return "tool_call"
        return "knowledge_qa"

    def _retrieve(self, question: str) -> list[RetrievalHit]:
        return self.vector_store.search(question, top_k=settings.top_k)

    def _plan_tools(self, question: str) -> list[ToolCall]:
        # 工具调用只输出结构化参数，后端再按白名单执行，避免模型直接操作数据库。
        company_name = _extract_company_name(question)
        calls = [ToolCall(tool="search_customer", args={"company_name": company_name})]
        if "邮件" in question or "跟进" in question:
            calls.append(ToolCall(tool="draft_followup_email", args={"company_name": company_name, "topic": "客户跟进"}))
        return calls

    def _execute_tools(self, calls: list[ToolCall]) -> list[ToolResult]:
        return [self.tools.execute(call) for call in calls]

    def _generate_answer(
        self,
        question: str,
        route: RouteType,
        hits: list[RetrievalHit],
        citations: list[Citation],
        tool_results: list[ToolResult],
    ) -> str:
        evidence = _format_evidence(hits)
        tools = json.dumps([asdict(result) for result in tool_results], ensure_ascii=False, indent=2)
        business_resources = self.resource_loader.load("agent_business_rules.md")
        prompt = self.prompt_manager.render(
            "agent_answer.md",
            PromptRenderInput(
                question=question,
                route=route,
                evidence=evidence or "未检索到可用知识库证据。",
                tool_results=tools or "无工具结果。",
                business_resources=business_resources,
            ),
        )
        if self.llm.enabled:
            try:
                return self.llm.complete(
                    [
                        LlmMessage(role="system", content="你负责生成可靠、可审计的企业 Agent 答案。"),
                        LlmMessage(role="user", content=prompt),
                    ]
                )
            except RuntimeError:
                return _compose_local_answer(question, route, evidence, citations, tool_results)

        return _compose_local_answer(question, route, evidence, citations, tool_results)

    def _review(self, draft: str, citations: list[Citation], tool_results: list[ToolResult]) -> AgentReview:
        issues: list[str] = []
        if not draft.strip():
            issues.append("答案为空。")
        if citations and "引用来源" not in draft:
            issues.append("答案缺少用户可见的引用来源。")
        failed_tools = [result.tool for result in tool_results if not result.ok]
        if failed_tools:
            issues.append(f"工具调用失败：{', '.join(failed_tools)}。")

        passed = not issues
        return AgentReview(
            passed=passed,
            risk_level="low" if passed else "medium",
            issues=issues,
            final_answer=draft if passed else "当前结果未通过审核，系统已阻止返回未经校验的答案。",
        )

    def _refusal(
        self,
        question: str,
        route: RouteType,
        best_score: float,
        tool_results: list[ToolResult],
    ) -> AgentAnswer:
        final_answer = "当前知识库没有足够依据回答该问题。建议补充相关制度文档或联系管理员确认。"
        return AgentAnswer(
            trace_id=self.recorder.trace_id,
            question=question,
            route=route,
            answer=final_answer,
            citations=[],
            confidence=round(max(best_score, 0.0), 4),
            refused=True,
            tool_results=tool_results,
            review=AgentReview(passed=True, risk_level="low", issues=[], final_answer=final_answer),
        )

    def _record_answer(self, answer: AgentAnswer) -> None:
        with self.recorder.span("agent.finalize", {"refused": answer.refused}) as span:
            span.update(
                {
                    "route": answer.route,
                    "confidence": answer.confidence,
                    "citation_count": len(answer.citations),
                    "tool_count": len(answer.tool_results),
                    "review_passed": answer.review.passed if answer.review else None,
                }
            )
        self.recorder.flush()


def _citation_from_hit(hit: RetrievalHit) -> Citation:
    return Citation(
        document_id=hit.chunk.document_id,
        title=hit.chunk.title,
        chunk_id=hit.chunk.chunk_id,
        score=round(hit.score, 4),
        heading_path=list(hit.chunk.metadata.get("heading_path") or []),
        chunk_strategy=str(hit.chunk.metadata.get("chunk_strategy", "")),
    )


def _compose_local_answer(
    question: str,
    route: RouteType,
    evidence: str,
    citations: list[Citation],
    tool_results: list[ToolResult],
) -> str:
    parts = [f"根据当前 Agent 工作流对“{question}”的处理结果："]
    if evidence:
        parts.append(evidence)
    if tool_results:
        parts.append("工具结果：")
        for result in tool_results:
            parts.append(json.dumps(asdict(result), ensure_ascii=False))
    if citations:
        parts.append(_format_sources(citations))
    elif route == "tool_call":
        parts.append("本次为业务工具请求，未命中知识库引用。")
    return "\n\n".join(parts)


def _format_evidence(hits: list[RetrievalHit]) -> str:
    lines = []
    for index, hit in enumerate(hits, start=1):
        if hit.score <= 0:
            continue
        heading = " > ".join(hit.chunk.metadata.get("heading_path") or [])
        label = f"章节：{heading}；" if heading else ""
        lines.append(f"[{index}] {label}相似度：{hit.score:.4f}\n{hit.chunk.text}")
    return "\n\n".join(lines)


def _format_sources(citations: list[Citation]) -> str:
    lines = ["引用来源："]
    for index, citation in enumerate(citations, start=1):
        heading = " > ".join(citation.heading_path) if citation.heading_path else "未标注章节"
        lines.append(
            f"{index}. 文档：{citation.title}；章节：{heading}；"
            f"片段：{citation.chunk_id}；相似度：{citation.score:.4f}"
        )
    return "\n".join(lines)


def _extract_company_name(question: str) -> str:
    for marker in ("公司", "客户"):
        index = question.find(marker)
        if index > 0:
            start = max(0, index - 12)
            return question[start : index + len(marker)].strip(" ，,。")
    return "演示客户"


def _compact(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def _summarize(value: object) -> object:
    if isinstance(value, list):
        return {"type": "list", "count": len(value)}
    if isinstance(value, str):
        return _compact(value)
    return str(value)


def agent_answer_to_dict(answer: AgentAnswer) -> dict[str, object]:
    return asdict(answer)
