from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Callable, Protocol, TypeVar

from ops_agent.config import settings
from ops_agent.models import AgentAnswer, AgentReview, Citation, RetrievalHit, ToolCall, ToolResult
from ops_agent.models.domain import RouteType
from ops_agent.prompts import PromptManager, PromptRenderInput
from ops_agent.resources import ResourceLoader
from ops_agent.services.llm_client import DeepSeekChatClient, LlmMessage
from ops_agent.services.rerank_service import Reranker, create_reranker
from ops_agent.services.tool_service import ToolRegistry
from ops_agent.services.trace_service import TraceRecorder
from ops_agent.services.vector_store import LocalVectorStore, PgVectorStore, create_vector_store

T = TypeVar("T")

REACT_SYSTEM_PROMPT = """你是 OpsAgent 的 ReAct 编排器。
你需要按 Thought -> Action -> Observation -> Reflection 的方式推进任务。

可用 Action：
- retrieve_knowledge: {"query": "..."}
- search_customer: {"company_name": "..."}
- draft_followup_email: {"company_name": "...", "topic": "..."}
- create_ticket: {"company_name": "...", "title": "...", "priority": "low|medium|high"}
- summarize_customer_visit: {"company_name": "...", "notes": "..."}
- final_answer: {"reason": "..."}

输出必须严格使用以下格式：
Thought: 当前判断
Action: action_name
Action Input: JSON对象

Few-shot 示例：
Thought: 用户询问高级客户售后响应时间，需要先检索知识库。
Action: retrieve_knowledge
Action Input: {"query":"高级客户售后响应时间"}

Observation: 检索到售后政策片段，说明高级客户响应时间为4小时。
Thought: 已经有足够证据，可以生成最终答案。
Action: final_answer
Action Input: {"reason":"已找到明确知识库依据"}
"""


class ChatClient(Protocol):
    enabled: bool

    def complete(self, messages: list[LlmMessage], temperature: float = 0.2) -> str:
        ...


@dataclass(frozen=True)
class AgentRunCommand:
    question: str


@dataclass
class AgentState:
    question: str
    route: RouteType = "knowledge_qa"
    step: int = 0
    thoughts: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    hits: list[RetrievalHit] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    final_ready: bool = False
    draft: str = ""
    review: AgentReview | None = None
    best_score: float = 0.0


class AgentService:
    """Explicit ReAct-style state machine for controllable Agent orchestration."""

    def __init__(
        self,
        vector_store: LocalVectorStore | PgVectorStore | None = None,
        recorder: TraceRecorder | None = None,
        llm: ChatClient | None = None,
        tools: ToolRegistry | None = None,
        prompt_manager: PromptManager | None = None,
        resource_loader: ResourceLoader | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.vector_store = vector_store or create_vector_store()
        self.recorder = recorder or TraceRecorder()
        self.llm = llm or DeepSeekChatClient()
        self.tools = tools or ToolRegistry()
        self.prompt_manager = prompt_manager or PromptManager()
        self.resource_loader = resource_loader or ResourceLoader()
        self.reranker = reranker or create_reranker()

    def run(self, command: AgentRunCommand | str) -> AgentAnswer:
        question = command if isinstance(command, str) else command.question
        question = question.strip()
        state = AgentState(question=question)
        self._validate_input(state)
        self._run_state_machine(state)

        if state.route == "knowledge_qa" and state.best_score < settings.min_relevance_score:
            answer = self._refusal(state)
            self._record_answer(answer)
            return answer

        state.draft = self._run_node("react.answer", lambda: self._generate_answer(state))
        state.review = self._run_node("react.review", lambda: self._review(state.draft, state.citations, state.tool_results))

        if not state.review.passed:
            answer = AgentAnswer(
                trace_id=self.recorder.trace_id,
                question=state.question,
                route=state.route,
                answer=state.review.final_answer or "当前结果未通过审核，请补充依据或稍后重试。",
                citations=state.citations,
                confidence=round(state.best_score, 4),
                refused=True,
                tool_results=state.tool_results,
                review=state.review,
            )
            self._record_answer(answer)
            return answer

        answer = AgentAnswer(
            trace_id=self.recorder.trace_id,
            question=state.question,
            route=state.route,
            answer=state.review.final_answer,
            citations=state.citations,
            confidence=round(state.best_score, 4),
            refused=False,
            tool_results=state.tool_results,
            review=state.review,
        )
        self._record_answer(answer)
        return answer

    def _validate_input(self, state: AgentState) -> None:
        with self.recorder.span("react.input.validate", {"question": _compact(state.question)}) as span:
            if not state.question:
                raise ValueError("问题不能为空。")
            span.update({"characters": len(state.question)})

    def _run_state_machine(self, state: AgentState) -> None:
        max_steps = max(settings.agent_max_retries + 3, 4)
        while not state.final_ready and state.step < max_steps:
            state.step += 1
            self._run_node("react.analyze", lambda: self._analyze(state))
            self._run_node("react.plan", lambda: self._plan(state))
            action = self._run_node("react.decide_action", lambda: self._decide_action(state))
            observation = self._run_node("react.act_observe", lambda: self._act_and_observe(state, action))
            state.observations.append(observation)
            self._run_node("react.reflect", lambda: self._reflect(state))
        if not state.final_ready:
            state.final_ready = True
            self.recorder.record(
                "react.stop",
                output_summary={"reason": "max_steps_reached", "steps": state.step},
            )

    def _analyze(self, state: AgentState) -> None:
        state.route = self._route(state.question)
        thought = f"问题路由为 {state.route}，需要结合知识检索和必要工具结果判断是否可回答。"
        state.thoughts.append(thought)
        self.recorder.record("react.thought", output_summary={"thought": thought, "route": state.route})

    def _plan(self, state: AgentState) -> None:
        plan = ["检索知识库证据"]
        if state.route in {"tool_call", "hybrid"}:
            plan.append("执行白名单业务工具")
        plan.append("根据观察结果反思是否足够回答")
        state.plan = plan
        self.recorder.record("react.plan.detail", output_summary={"plan": plan})

    def _decide_action(self, state: AgentState) -> ToolCall:
        if self.llm.enabled:
            maybe_action = self._llm_decide_action(state)
            if maybe_action is not None:
                return maybe_action
        if state.route in {"tool_call", "hybrid"} and not state.tool_results:
            calls = self._plan_tools(state.question)
            state.tool_calls = calls
            return calls[0] if calls else ToolCall(tool="final_answer", args={"reason": "no_tool_required"})
        if not state.hits:
            return ToolCall(tool="retrieve_knowledge", args={"query": state.question})
        return ToolCall(tool="final_answer", args={"reason": "enough_observation"})

    def _llm_decide_action(self, state: AgentState) -> ToolCall | None:
        messages = [
            LlmMessage(role="system", content=REACT_SYSTEM_PROMPT),
            LlmMessage(role="user", content=self._format_react_context(state)),
        ]
        self.recorder.record_llm_prompt("llm.react.prompt", [asdict(message) for message in messages])
        try:
            raw = self.llm.complete(messages, temperature=0.1)
        except RuntimeError as exc:
            self.recorder.record("llm.react.error", output_summary={"error": str(exc)})
            return None
        self.recorder.record_llm_raw_output("llm.react.raw_output", raw)
        parsed = _parse_action(raw)
        if parsed is None:
            self.recorder.record_parse_failure("llm.react.parse_failure", raw, "Action or Action Input not found.")
            return None
        return parsed

    def _format_react_context(self, state: AgentState) -> str:
        return (
            f"用户问题：{state.question}\n"
            f"当前路由：{state.route}\n"
            f"计划：{json.dumps(state.plan, ensure_ascii=False)}\n"
            f"历史 Thought：{json.dumps(state.thoughts, ensure_ascii=False)}\n"
            f"Observation：{json.dumps(state.observations, ensure_ascii=False)}\n"
            f"已执行工具：{json.dumps([asdict(result) for result in state.tool_results], ensure_ascii=False)}\n"
            "请给出下一步 Thought/Action/Action Input。"
        )

    def _act_and_observe(self, state: AgentState, action: ToolCall) -> str:
        if action.tool == "retrieve_knowledge":
            query = str(action.args.get("query") or state.question)
            recalled_hits = self.vector_store.search(query, top_k=settings.retrieval_top_k)
            hits = self.reranker.rerank(query, recalled_hits, top_k=settings.rerank_top_k)
            state.hits = hits
            state.citations = [_citation_from_hit(hit) for hit in hits if hit.score > 0]
            state.best_score = hits[0].score if hits else 0.0
            observation = (
                f"初步召回 {len(recalled_hits)} 个知识片段，"
                f"重排后保留 {len(hits)} 个，最高相关度 {state.best_score:.4f}。"
            )
            self.recorder.record_tool_io(
                "retrieve_knowledge",
                {"query": query, "retrieval_top_k": settings.retrieval_top_k, "rerank_top_k": settings.rerank_top_k},
                {
                    "recalled_count": len(recalled_hits),
                    "hit_count": len(hits),
                    "best_score": round(state.best_score, 4),
                    "recalled_chunk_ids": [hit.chunk.chunk_id for hit in recalled_hits],
                    "chunk_ids": [hit.chunk.chunk_id for hit in hits],
                },
            )
            return observation

        if action.tool == "final_answer":
            state.final_ready = True
            return f"进入最终回答阶段：{action.args.get('reason', 'ready')}"

        if action.tool not in self.tools.names:
            result = ToolResult(tool=action.tool, ok=False, error="Tool is not whitelisted.")
        else:
            result = self.tools.execute(action)
        state.tool_results.append(result)
        self.recorder.record_tool_io(action.tool, action.args, asdict(result))

        if state.route in {"tool_call", "hybrid"}:
            planned = state.tool_calls or self._plan_tools(state.question)
            state.tool_calls = planned
            executed = {result.tool for result in state.tool_results}
            remaining = [call for call in planned if call.tool not in executed]
            if remaining:
                next_result = self._act_and_observe(state, remaining[0])
                return f"工具 {action.tool} 已执行；{next_result}"
        return f"工具 {action.tool} 返回 {'成功' if result.ok else '失败'}。"

    def _reflect(self, state: AgentState) -> None:
        has_evidence = state.best_score >= settings.min_relevance_score
        tool_ready = state.route == "knowledge_qa" or bool(state.tool_results) or state.route == "hybrid"
        if state.route == "tool_call":
            tool_ready = bool(state.tool_results)
        state.final_ready = state.final_ready or (has_evidence and tool_ready)
        reflection = {
            "has_evidence": has_evidence,
            "tool_ready": tool_ready,
            "final_ready": state.final_ready,
            "best_score": round(state.best_score, 4),
        }
        self.recorder.record("react.reflection", output_summary=reflection)

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
        knowledge_keywords = ("政策", "规则", "制度", "知识库", "售后", "响应", "鏀跨瓥", "瑙勫垯", "鍒跺害", "鐭ヨ瘑搴", "鍞悗", "鍝嶅簲")
        wants_tool = (
            any(keyword in lowered for keyword in ("crm", "邮件", "跟进", "工单", "拜访", "纪要", "閭欢", "璺熻繘", "宸ュ崟", "鎷滆", "绾"))
            or ("查询" in lowered and "客户" in lowered)
            or ("鏌ヨ" in lowered and "瀹㈡埛" in lowered)
        )
        wants_knowledge = any(keyword in lowered for keyword in knowledge_keywords)
        if wants_tool and wants_knowledge:
            return "hybrid"
        if wants_tool:
            return "tool_call"
        return "knowledge_qa"

    def _plan_tools(self, question: str) -> list[ToolCall]:
        company_name = _extract_company_name(question)
        calls = [ToolCall(tool="search_customer", args={"company_name": company_name})]
        if any(keyword in question for keyword in ("邮件", "跟进", "閭欢", "璺熻繘")):
            calls.append(ToolCall(tool="draft_followup_email", args={"company_name": company_name, "topic": "客户跟进"}))
        if any(keyword in question for keyword in ("工单", "故障", "问题", "宸ュ崟", "鏁呴殰", "闂")):
            calls.append(
                ToolCall(
                    tool="create_ticket",
                    args={
                        "company_name": company_name,
                        "title": _extract_ticket_title(question),
                        "priority": _infer_priority(question),
                    },
                )
            )
        if any(keyword in question for keyword in ("拜访", "纪要", "会议", "鎷滆", "绾", "浼氳")):
            calls.append(
                ToolCall(
                    tool="summarize_customer_visit",
                    args={"company_name": company_name, "notes": _extract_visit_notes(question)},
                )
            )
        return calls

    def _generate_answer(self, state: AgentState) -> str:
        evidence = _format_evidence(state.hits)
        tools = json.dumps([asdict(result) for result in state.tool_results], ensure_ascii=False, indent=2)
        business_resources = self.resource_loader.load("agent_business_rules.md")
        prompt = self.prompt_manager.render(
            "agent_answer.md",
            PromptRenderInput(
                question=state.question,
                route=state.route,
                evidence=evidence or "未检索到可用知识库证据。",
                tool_results=tools or "无工具结果。",
                business_resources=business_resources,
            ),
        )
        messages = [
            LlmMessage(role="system", content="你负责生成可靠、可审计的企业 Agent 答案。"),
            LlmMessage(role="user", content=prompt),
        ]
        self.recorder.record_llm_prompt("llm.answer.prompt", [asdict(message) for message in messages])
        if self.llm.enabled:
            try:
                raw = self.llm.complete(messages)
                self.recorder.record_llm_raw_output("llm.answer.raw_output", raw)
                return raw
            except RuntimeError as exc:
                self.recorder.record("llm.answer.error", output_summary={"error": str(exc)})
        return _compose_local_answer(state.question, state.route, evidence, state.citations, state.tool_results)

    def _review(self, draft: str, citations: list[Citation], tool_results: list[ToolResult]) -> AgentReview:
        issues: list[str] = []
        if not draft.strip():
            issues.append("答案为空。")
        if citations and "引用来源" not in draft and "寮曠敤鏉ユ簮" not in draft:
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

    def _refusal(self, state: AgentState) -> AgentAnswer:
        final_answer = "当前知识库没有足够依据回答该问题。建议补充相关制度文档或联系管理员确认。"
        return AgentAnswer(
            trace_id=self.recorder.trace_id,
            question=state.question,
            route=state.route,
            answer=final_answer,
            citations=[],
            confidence=round(max(state.best_score, 0.0), 4),
            refused=True,
            tool_results=state.tool_results,
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


def _parse_action(raw: str) -> ToolCall | None:
    action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", raw)
    input_match = re.search(r"Action Input:\s*(\{.*\})", raw, flags=re.DOTALL)
    if not action_match or not input_match:
        return None
    try:
        args = json.loads(input_match.group(1).strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(args, dict):
        return None
    return ToolCall(tool=action_match.group(1).strip(), args=args)


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
    lines = ["引用来源：", "寮曠敤鏉ユ簮："]
    for index, citation in enumerate(citations, start=1):
        heading = " > ".join(citation.heading_path) if citation.heading_path else "未标注章节"
        lines.append(
            f"{index}. 文档：{citation.title}；章节：{heading}；"
            f"片段：{citation.chunk_id}；相似度：{citation.score:.4f}"
        )
    return "\n".join(lines)


def _extract_company_name(question: str) -> str:
    for marker in ("公司", "客户", "鍏徃", "瀹㈡埛"):
        index = question.find(marker)
        if index > 0:
            start = max(0, index - 12)
            return question[start : index + len(marker)].strip(" ，。")
    return "演示客户"


def _extract_ticket_title(question: str) -> str:
    for marker in ("问题", "故障", "工单", "闂", "鏁呴殰", "宸ュ崟"):
        index = question.find(marker)
        if index >= 0:
            start = max(0, index - 18)
            end = min(len(question), index + len(marker) + 18)
            return question[start:end].strip(" ，。")
    return "客户问题跟进"


def _extract_visit_notes(question: str) -> str:
    for marker in ("拜访", "纪要", "会议", "鎷滆", "绾", "浼氳"):
        index = question.find(marker)
        if index >= 0:
            return question[max(0, index - 20) :].strip(" ，。")
    return question[:80].strip(" ，。")


def _infer_priority(question: str) -> str:
    if any(keyword in question for keyword in ("紧急", "严重", "阻塞", "高优先级", "绱ф€", "涓ラ噸", "闃诲", "楂樹紭鍏堢骇")):
        return "high"
    if any(keyword in question for keyword in ("低优先级", "不紧急", "浣庝紭鍏堢骇", "涓嶇揣鎬")):
        return "low"
    return "medium"


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
