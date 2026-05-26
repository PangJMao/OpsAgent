from __future__ import annotations

from dataclasses import dataclass, field
import re

from ops_agent.config.retrieval_config import DEFAULT_RETRIEVAL_CONFIG
from ops_agent.models import RetrievalHit
from ops_agent.services.business_scene import BusinessFrame, parse_business_frame, stage_bucket
from ops_agent.services.rerank_service import Reranker
from ops_agent.services.retriever.hybrid_retriever import HybridRetriever
from ops_agent.services.retriever.keyword_retriever import KeywordRetriever
from ops_agent.services.retriever.schema import CompressedContext
from ops_agent.services.retriever.vector_retriever import VectorRetriever


@dataclass(frozen=True)
class SourceRef:
    doc_name: str
    section: str = ""
    sheet_name: str = ""
    page: int | None = None
    row: int | None = None
    topic: str = ""


@dataclass
class RagWorkflowState:
    question: str
    question_type: str = "knowledge_qa"
    risk_level: str = "low"
    queries: list[str] = field(default_factory=list)
    recalled_hits: list[RetrievalHit] = field(default_factory=list)
    reranked_hits: list[RetrievalHit] = field(default_factory=list)
    final_hits: list[RetrievalHit] = field(default_factory=list)
    contexts: list[CompressedContext] = field(default_factory=list)
    context: str = ""
    sources: list[SourceRef] = field(default_factory=list)
    confidence: float = 0.0
    confidence_band: str = "low"
    quality_flags: list[str] = field(default_factory=list)
    should_answer: bool = False
    needs_clarification: bool = False
    business_frame: BusinessFrame | None = None
    debug: dict[str, object] = field(default_factory=dict)


class EnterpriseRagWorkflow:
    """LangGraph-ready RAG workflow facade."""

    def __init__(self, store, reranker: Reranker) -> None:
        self.hybrid = HybridRetriever(
            vector_retriever=VectorRetriever(store),
            keyword_retriever=KeywordRetriever(store),
            reranker=reranker,
            config=DEFAULT_RETRIEVAL_CONFIG,
        )

    def run(self, question: str) -> RagWorkflowState:
        state = RagWorkflowState(question=question.strip())
        self.classify_question(state)
        state.business_frame = parse_business_frame(state.question, state.question_type)
        result = self.hybrid.retrieve(state.question, question_type=state.question_type)
        filtered_hits = _filter_scene_hits(result.hits, state.business_frame)
        filtered_contexts = _filter_scene_contexts(result.contexts, state.business_frame)
        state.queries = list(result.debug.get("rewritten_queries", []))
        state.recalled_hits = []
        state.reranked_hits = []
        state.final_hits = filtered_hits
        state.contexts = filtered_contexts
        state.context = _contexts_to_text(filtered_contexts)
        state.sources = dedupe_sources([_source_from_context(context) for context in filtered_contexts])
        state.confidence_band = result.confidence
        state.confidence = _confidence_value(result.confidence, result.hits)
        state.should_answer = result.can_answer
        state.needs_clarification = not result.can_answer
        quality = result.debug.get("quality", {})
        state.quality_flags = list(quality.get("flags", [])) if isinstance(quality, dict) else []
        if any(context.risk_level == "high" for context in filtered_contexts):
            state.risk_level = "high"
        state.debug = {
            **result.debug,
            "business_frame": state.business_frame.to_dict(),
            "scene_filter": {
                "input_hits": len(result.hits),
                "output_hits": len(filtered_hits),
                "input_contexts": len(result.contexts),
                "output_contexts": len(filtered_contexts),
            },
        }
        return state

    def classify_question(self, state: RagWorkflowState) -> None:
        text = state.question
        lower = text.lower()

        if _contains(text, "多次跳票", "跳票", "认可鼓励"):
            state.question_type = "recognition_encouragement"
        elif _contains(text, "文明用语", "有效安抚", "安抚"):
            state.question_type = "effective_comfort"
            state.risk_level = "medium"
        elif _contains(text, "自报家门", "开场", "结束语", "非首通", "首通电话"):
            state.question_type = "call_protocol"
        elif _contains(text, "过两天", "约定还款时间", "约定时间", "还款时间"):
            state.question_type = "repayment_appointment"
        elif _contains(text, "先还一部分", "部分金额", "部分还款"):
            state.question_type = "partial_repayment"
        elif _contains(text, "封闭式", "开放式", "拒绝回答", "继续追问", "现实困难", "还款计划"):
            state.question_type = "funding_followup"
            state.risk_level = "medium"
        elif _contains(text, "零容忍"):
            state.question_type = "zero_tolerance"
            state.risk_level = "high"
        elif _contains(text, "三方号码", "何时办理", "何地办理", "是否停机", "无法报姓名", "不认识客户"):
            state.question_type = "third_party_identity_check"
            state.risk_level = "medium"
        elif _contains(text, "核资", "还款能力", "资金来源", "收入", "工资", "工作单位", "可调配资金", "发放时间", "收入来源", "原单位"):
            state.question_type = "funding_check"
            state.risk_level = "high" if stage_bucket(text) == "D10-D15" else "medium"
        elif _contains(text, "房产", "车产", "存款", "股票", "基金", "资产"):
            state.question_type = "asset_inquiry"
            state.risk_level = "high"
        elif _contains(text, "冻结", "法院", "诉讼", "律所", "移交", "律师函", "法务", "起诉", "法律责任", "强制执行", "调解"):
            state.question_type = "legal_compliance"
            state.risk_level = "high"
        elif _contains(text, "国外", "境外"):
            state.question_type = "clarification_overseas"
            state.risk_level = "medium"
        elif _contains(text, "医院", "住院", "看病"):
            state.question_type = "clarification_medical"
            state.risk_level = "medium"
        elif _contains(text, "紧急联系人", "预留联系人", "联系人", "不想被联系", "新的手机号", "家人", "亲属"):
            state.question_type = "contact_boundary"
            state.risk_level = "high" if _contains(text, "停止", "不想", "手机号", "家人") else "medium"
        elif _contains(text, "不实承诺", "直接答应", "可以直接答应", "承诺"):
            state.question_type = "false_commitment"
            state.risk_level = "high"
        elif _contains(text, "骚扰", "投诉", "我要投诉"):
            state.question_type = "complaint_threat"
            state.risk_level = "high"
        elif _contains(text, "骂人", "辱骂", "骂"):
            state.question_type = "customer_abuse"
            state.risk_level = "medium"
        elif _stage_from_question(text) or stage_bucket(text):
            state.question_type = "stage_script"
            state.risk_level = "medium"
        elif _contains(text, "新人", "刚开始", "新手"):
            state.question_type = "new_collector"
        elif _contains(text, "逾期天数下降", "历史最高逾期", "谈判点"):
            state.question_type = "negotiation_stage"
            state.risk_level = "medium"
        elif _contains(text, "减免", "利息", "优惠"):
            state.question_type = "reduction_request"
            state.risk_level = "medium"
        elif _contains(text, "沟通", "话术", "催收") or (
            "客户" in text and _contains(text, "技巧", "骂人", "投诉", "联系人", "新人")
        ):
            state.question_type = "communication_script"
        elif _contains(text, "怎么做", "如何处理", "步骤", "流程", "操作"):
            state.question_type = "operation"
        elif _contains(text, "总结", "有哪些", "归纳", "要点"):
            state.question_type = "summary"

        if "d10" in lower or "d15" in lower or stage_bucket(text) == "D10-D15":
            state.risk_level = "high"


def build_source_text(sources: list[SourceRef]) -> str:
    sources = dedupe_sources(sources)
    if not sources:
        return "引用来源：无"
    lines = ["引用来源："]
    for index, source in enumerate(sources[:5], start=1):
        parts = [f"《{source.doc_name}》"]
        if source.sheet_name:
            parts.append(f"Sheet：{source.sheet_name}")
        if source.section:
            parts.append(f"章节：{source.section}")
        if source.page:
            parts.append(f"第 {source.page} 页")
        if source.row:
            parts.append(f"第 {source.row} 行")
        if source.topic:
            parts.append(source.topic)
        lines.append(f"{index}. {' '.join(parts)}")
    return "\n".join(lines)


def build_answer_prompt(state: RagWorkflowState) -> str:
    type_instruction = {
        "communication_script": "请总结为客户沟通技巧，覆盖身份说明、阶段化沟通、语气、安抚、认可鼓励、不实承诺边界。",
        "new_collector": "请面向新人给出入门沟通注意事项，强调核身、边界、记录和升级。",
        "customer_abuse": "请回答客户辱骂场景，包含先安抚、不回怼、必要时升级或结束通话。",
        "complaint_threat": "请回答投诉/骚扰指控场景，包含安抚、核对联系频次、避免刺激、升级处理。",
        "stage_script": "请按逾期阶段回答，区分可加强提醒和禁止威胁、不实承诺。",
        "contact_boundary": "请按联系人沟通边界回答，说明哪些信息不能透露、何时停止或转人工核对。",
        "asset_inquiry": "请按敏感资产信息问题回答，优先说明是否允许以及合规边界。",
        "funding_check": "请按核资/还款能力确认问题回答，区分阶段、可询问方向、禁止问法和合规边界。",
        "legal_compliance": "请按法务/合规高风险问题回答，包含风险提示、可用说法、不建议说法、人工核对项。",
        "false_commitment": "请解释不实承诺并列举催收沟通风险点。",
        "negotiation_stage": "请说明谈判点应基于当前阶段和系统记录，不要用过期或误导性依据。",
        "reduction_request": "请说明减免类问题必须以政策、权限和系统记录为准，不要自行承诺。",
        "clarification_medical": "请在证据不足时给出保守、人文关怀且需澄清的答法。",
        "clarification_overseas": "请在证据不足时说明联系人/家人联系边界和需要核对的信息。",
        "operation": "请按操作型问题回答，包含推荐处理方式、操作步骤、禁止事项。",
        "summary": "请按总结型问题回答，包含结论、关键要点、注意事项。",
    }.get(state.question_type, "请按普通知识问答回答。")
    frame = state.business_frame.to_dict() if state.business_frame else {}
    return f"""用户问题：{state.question}
问题类型：{state.question_type}
风险等级：{state.risk_level}
置信度：{state.confidence_band}
场景拆解：{frame}

结构化上下文：
{state.context}

回答要求：{type_instruction}
- 必须基于上下文归纳，不要直接粘贴原始 chunk。
- 不要输出 chunk_id、hybrid_score、vector_score、rerank_score、数据库 UUID。
- 如证据不足，请明确说明“根据当前知识库暂无法确认”，并列出需要核对的资料。
- 末尾使用业务友好的引用来源。
{build_source_text(state.sources)}"""


def dedupe_sources(sources: list[SourceRef]) -> list[SourceRef]:
    result: list[SourceRef] = []
    seen: set[tuple[str, str, str, str]] = set()
    for source in sources:
        key = (source.doc_name, source.sheet_name, source.section, source.topic)
        if key in seen:
            continue
        seen.add(key)
        result.append(source)
    return result


def _contexts_to_text(contexts: list[CompressedContext]) -> str:
    blocks = []
    for index, context in enumerate(contexts, start=1):
        points = "\n".join(f"- {point}" for point in context.key_points)
        blocks.append(
            f"证据 {index}\n"
            f"来源：{context.source}\n"
            f"主题：{context.topic}\n"
            f"风险：{context.risk_level}\n"
            f"要点：\n{points}"
        )
    return "\n\n".join(blocks)


def _source_from_context(context: CompressedContext) -> SourceRef:
    label = context.source
    doc_name = label
    match = re.search(r"《([^》]+)》", label)
    if match:
        doc_name = match.group(1)
    elif "《" in label:
        doc_name = label.split("《", 1)[-1].rstrip("》")
    sheet = _after_any(label, ("Sheet：",))
    section = _after_any(label, ("章节：",))
    return SourceRef(doc_name=doc_name, sheet_name=sheet, section=section, topic=context.topic)


def _after_any(text: str, markers: tuple[str, ...]) -> str:
    for marker in markers:
        if marker in text:
            value = text.split(marker, 1)[1]
            for next_marker in (" Sheet：", " 章节：", " 第", " 页", " 行"):
                if next_marker.strip() != marker.strip() and next_marker in value:
                    value = value.split(next_marker, 1)[0]
            return value.strip()
    return ""


def _confidence_value(confidence: str, hits: list[RetrievalHit]) -> float:
    if hits:
        return round(max(hit.score for hit in hits), 4)
    return {"high": 0.9, "medium": 0.65, "low": 0.0}.get(confidence, 0.0)


def _filter_scene_hits(hits: list[RetrievalHit], frame: BusinessFrame | None) -> list[RetrievalHit]:
    if frame is None:
        return hits
    filtered = [hit for hit in hits if _hit_matches_scene(hit, frame)]
    return filtered or hits[:1]


def _filter_scene_contexts(contexts: list[CompressedContext], frame: BusinessFrame | None) -> list[CompressedContext]:
    if frame is None:
        return contexts
    filtered: list[CompressedContext] = []
    for context in contexts:
        points = [point for point in context.key_points if _point_matches_scene(point, frame)]
        if not points:
            continue
        filtered.append(
            CompressedContext(
                source=context.source,
                topic=context.topic,
                key_points=points,
                risk_level=context.risk_level,
            )
        )
    return filtered or contexts[:1]


def _hit_matches_scene(hit: RetrievalHit, frame: BusinessFrame) -> bool:
    text = f"{hit.chunk.text} {hit.chunk.metadata}"
    if _looks_like_noise(text):
        return False
    if not _stage_matches(text, frame.stage):
        return False
    if frame.subject == "联系人" and _contains(text, "房产", "车产", "存款", "股票", "基金") and not _contains(text, "不得", "不要", "不应"):
        return False
    return True


def _point_matches_scene(point: str, frame: BusinessFrame) -> bool:
    if _looks_like_noise(point):
        return False
    if not _stage_matches(point, frame.stage):
        return False
    if frame.subject == "联系人" and _contains(point, "房产", "车产", "存款", "股票", "基金") and not _contains(point, "不得", "不要", "不应"):
        return False
    return True


def _stage_matches(text: str, stage: str) -> bool:
    if not stage:
        return True
    other_stage_markers = {
        "D4-D6": ("D10-D15", "D10", "D11", "D12", "D13", "D14", "D15"),
        "D7-D9": ("D4-D6", "D10-D15"),
        "D10-D15": ("D4-D6", "D7-D9"),
    }.get(stage, ())
    if any(marker in text.upper() for marker in other_stage_markers):
        return stage in text.upper()
    return True


def _looks_like_noise(text: str) -> bool:
    markers = (
        "column_",
        "为本人/联系人",
        "| |",
        "Root Entry",
        "SummaryInformation",
        "DocumentSummaryInformation",
        "WordDocument",
        "\ufffd",
        "核对三方号码",
        "何时办理",
        "何地办理",
        "使用期间是否停机",
        "用户画像",
        "当天非首通电话",
        "自报家门",
        "主动挂机",
        "参考话术:请问",
        "张三",
        "李四",
    )
    return any(marker in text for marker in markers)


def _contains(text: str, *terms: str) -> bool:
    return any(term and term in text for term in terms)


def _stage_from_question(question: str) -> str:
    match = re.search(r"D\d+\s*[-~至到]\s*D?\d+", question, flags=re.IGNORECASE)
    return match.group(0) if match else ""
