from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re

from ops_agent.models import Citation, RagAnswer
from ops_agent.services.document_service import (
    chunk_document,
    load_text_document,
    persist_normalized_markdown,
    persist_source_document,
)
from ops_agent.services.document_processing.pipeline import IngestionPipeline
from ops_agent.services.document_processing.vector_store import LocalDocumentVectorStore, PgDocumentVectorStore
from ops_agent.services.business_rule_service import compose_business_rule_answer
from ops_agent.services.llm_client import DeepSeekChatClient, LlmMessage
from ops_agent.services.rag_workflow import RagWorkflowState, EnterpriseRagWorkflow, build_answer_prompt, build_source_text
from ops_agent.services.rerank_service import Reranker, create_reranker
from ops_agent.services.trace_service import TraceRecorder
from ops_agent.services.vector_store import LocalVectorStore, PgVectorStore, create_vector_store

SYSTEM_PROMPT = """你是企业知识库 Agent 的回答生成器。

必须遵守：
1. 只基于结构化证据回答，不编造知识库没有的制度、承诺、数字或责任人。
2. 不要堆砌原始片段，要把证据归纳成业务用户能理解的结论和方法。
3. 不要输出乱码、chunk_id、相似度分数、数据库 ID、embedding 信息或 debug 信息。
4. 法务、诉讼、征信、威胁、联系人边界等高风险内容必须加风险提示。
5. 证据中等相关时，使用“根据当前知识库可归纳为”，避免绝对化表达。
6. 末尾只展示业务友好的来源：文档名、章节、Sheet、页码或行号。"""


class RagService:
    """RAG application service: document ingestion and workflow-based question answering."""

    def __init__(
        self,
        vector_store: LocalVectorStore | PgVectorStore | None = None,
        recorder: TraceRecorder | None = None,
        llm: DeepSeekChatClient | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.vector_store = vector_store or create_vector_store()
        self.recorder = recorder or TraceRecorder()
        self.llm = llm or DeepSeekChatClient()
        self.reranker = reranker or create_reranker()
        self.workflow = EnterpriseRagWorkflow(store=self.vector_store, reranker=self.reranker)

    def ingest(self, path: Path) -> dict[str, object]:
        if path.suffix.lower() in {".doc", ".docx", ".xlsx", ".csv", ".pdf"}:
            return self._ingest_with_processing_pipeline(path)

        with self.recorder.span("document.normalize", {"path": str(path)}) as span:
            document = load_text_document(path)
            span.update(
                {
                    "document_id": document.document_id,
                    "characters": len(document.content),
                    "source_format": document.metadata.get("source_format"),
                    "normalized_format": document.metadata.get("normalized_format"),
                }
            )

        with self.recorder.span("document.persist", {"document_id": document.document_id}) as span:
            stored_path = persist_source_document(path, document.document_id)
            normalized_path = persist_normalized_markdown(document)
            span.update({"stored_path": str(stored_path), "normalized_path": str(normalized_path)})

        with self.recorder.span("document.chunk", {"document_id": document.document_id}) as span:
            chunks = chunk_document(document)
            span.update({"chunk_count": len(chunks), "strategy_counts": _strategy_counts(chunks)})

        with self.recorder.span("vector_store.upsert", {"chunk_count": len(chunks)}) as span:
            self.vector_store.upsert_chunks(chunks)
            span.update({"total_chunks": self.vector_store.count()})

        trace_path = self.recorder.flush()
        return {
            "trace_id": self.recorder.trace_id,
            "trace_path": str(trace_path),
            "document_id": document.document_id,
            "title": document.title,
            "chunk_count": len(chunks),
            "strategy_counts": _strategy_counts(chunks),
        }

    def _ingest_with_processing_pipeline(self, path: Path) -> dict[str, object]:
        adapter = (
            PgDocumentVectorStore(self.vector_store)
            if isinstance(self.vector_store, PgVectorStore)
            else LocalDocumentVectorStore(self.vector_store)  # type: ignore[arg-type]
        )
        pipeline = IngestionPipeline(vector_store=adapter)
        with self.recorder.span("document.processing.ingest", {"path": str(path)}) as span:
            result = pipeline.ingest_path(path, write=True)
            if result.failed:
                raise RuntimeError("; ".join(f"{source}: {error}" for source, error in result.failed.items()))
            span.update({"source_count": result.source_count, "chunk_count": result.chunk_count})
        trace_path = self.recorder.flush()
        first_chunk = result.chunks[0] if result.chunks else None
        return {
            "trace_id": self.recorder.trace_id,
            "trace_path": str(trace_path),
            "document_id": first_chunk.document_id if first_chunk else path.stem,
            "title": first_chunk.title if first_chunk else path.stem,
            "chunk_count": result.chunk_count,
            "strategy_counts": _ingested_strategy_counts(result.chunks),
        }

    def ask(self, question: str) -> RagAnswer:
        normalized_question = question.strip()
        with self.recorder.span("query.validate", {"question": _compact(normalized_question)}) as span:
            if not normalized_question:
                raise ValueError("问题不能为空。")
            span.update({"characters": len(normalized_question)})

        with self.recorder.span("rag.workflow", {"question": _compact(normalized_question)}) as span:
            state = self.workflow.run(normalized_question)
            span.update(
                {
                    "question_type": state.question_type,
                    "risk_level": state.risk_level,
                    "queries": state.queries,
                    "recalled_count": len(state.recalled_hits),
                    "reranked_count": len(state.reranked_hits),
                    "final_count": len(state.final_hits),
                    "confidence": state.confidence,
                    "confidence_band": state.confidence_band,
                    "quality_flags": state.quality_flags,
                }
            )

        if not state.should_answer:
            answer = RagAnswer(
                trace_id=self.recorder.trace_id,
                question=normalized_question,
                answer=_low_confidence_answer(state),
                citations=[],
                confidence=state.confidence,
                refused=True,
            )
            self._record_answer(answer)
            return answer

        with self.recorder.span("answer.compose", {"hit_count": len(state.final_hits)}) as span:
            answer_text = self._compose_answer(state)
            span.update({"source_count": len(state.sources), "answer_chars": len(answer_text)})

        answer = RagAnswer(
            trace_id=self.recorder.trace_id,
            question=normalized_question,
            answer=answer_text,
            citations=[_citation_from_state_hit(hit) for hit in state.final_hits],
            confidence=state.confidence,
            refused=False,
        )
        self._record_answer(answer)
        return answer

    def _compose_answer(self, state: RagWorkflowState) -> str:
        rule_answer = compose_business_rule_answer(state)
        if rule_answer:
            self.recorder.record("answer.business_rule", output_summary={"question_type": state.question_type})
            return rule_answer

        messages = [
            LlmMessage(role="system", content=SYSTEM_PROMPT),
            LlmMessage(role="user", content=build_answer_prompt(state)),
        ]
        self.recorder.record_llm_prompt("llm.rag.prompt", [asdict(message) for message in messages])
        if self.llm.enabled:
            try:
                raw = self.llm.complete(messages, temperature=0.2)
                self.recorder.record_llm_raw_output("llm.rag.raw_output", raw)
                return _sanitize_final_answer(raw, state)
            except RuntimeError as exc:
                self.recorder.record("llm.rag.error", output_summary={"error": str(exc)})
        return _compose_local_answer(state)

    def build_public_reasoning(self, question: str, answer: RagAnswer) -> str:
        if answer.refused:
            return "我检索了知识库，但当前证据不足以支撑可靠回答，因此没有强行生成结论。"
        source_names = []
        for citation in answer.citations:
            if citation.title and citation.title not in source_names:
                source_names.append(citation.title)
        source_text = "、".join(source_names[:3]) if source_names else "已检索到的知识库资料"
        return f"我围绕“{question.strip()}”做了多路检索、去重和证据压缩，主要依据来自：{source_text}。"

    def _record_answer(self, answer: RagAnswer) -> None:
        with self.recorder.span("answer.finalize", {"refused": answer.refused}) as span:
            span.update(
                {
                    "confidence": answer.confidence,
                    "citation_count": len(answer.citations),
                    "answer_chars": len(answer.answer),
                }
            )
        self.recorder.flush()


def _compose_local_answer(state: RagWorkflowState) -> str:
    sources = build_source_text(state.sources)
    prefix = "根据当前知识库可归纳为：" if state.confidence_band == "medium" else ""
    points = _bullet_points(state, limit=5)
    question_type = state.question_type

    if question_type == "recognition_encouragement":
        return (
            f"结论\n{prefix}客户多次跳票时，仍然需要做认可鼓励，但认可的是客户愿意接听、愿意沟通或仍有处理意愿，不能认可或纵容跳票行为本身。\n\n"
            f"关键要点\n{points}\n\n"
            f"{_guidance_section(state)}\n\n"
            "注意事项\n"
            "1. 不要机械夸奖，也不要说成客户跳票是合理的。\n"
            "2. 认可鼓励要具体，例如围绕愿意接电话、愿意说明困难、曾有正常还款记录等事实展开。\n"
            "3. 认可后要回到具体处理日期、金额和后续安排。\n\n"
            f"{sources}"
        )

    if question_type == "repayment_appointment":
        return (
            f"结论\n{prefix}约定还款时间不能只说“过两天”“过几天”这类模糊时间，应明确到今天、明天或具体日期，并同步确认金额和方式。\n\n"
            f"关键要点\n{points}\n\n"
            f"{_guidance_section(state)}\n\n"
            "注意事项\n"
            "1. 约定时间必须可记录、可复核，避免使用模糊表达。\n"
            "2. 如客户有异议处理，异议处理后需要再次确认还款时间。\n"
            "3. 同步确认还款金额、还款方式和后续跟进节点。\n\n"
            f"{sources}"
        )

    if question_type == "call_protocol":
        return (
            f"结论\n{prefix}开场应先表明机构身份和来意；当天非首通也需要再次自报家门，结束时应礼貌收口并说“再见”。\n\n"
            f"关键要点\n{points}\n\n"
            f"{_guidance_section(state)}\n\n"
            "注意事项\n"
            "1. 委外场景应说明受平台委托，不要只说平台名称造成身份不清。\n"
            "2. 电话接通后应尽早表明代表机构名称，核身和开场顺序可按场景调整。\n"
            "3. 客户主动挂机、信号异常或争执场景按录音情况特殊判断，但催收员自身仍应保持礼貌收口。\n\n"
            f"{sources}"
        )

    if question_type == "effective_comfort":
        return (
            f"结论\n{prefix}客户骂人或情绪升高时，仅说“请使用文明用语”通常不算有效安抚；有效安抚应先承接情绪，再引导回问题处理。\n\n"
            f"关键要点\n{points}\n\n"
            f"{_guidance_section(state)}\n\n"
            "注意事项\n"
            "1. 不回怼、不刺激、不用命令式表达激化矛盾。\n"
            "2. 如果客户持续辱骂且无法沟通，记录情况并按规则升级或结束。\n"
            "3. 安抚不能停留在重复套话，应回应客户具体不满点。\n\n"
            f"{sources}"
        )

    if question_type == "partial_repayment":
        return (
            f"结论\n{prefix}客户只能先还一部分时，应确认可处理金额、具体日期、剩余部分安排和后续跟进节点。\n\n"
            f"关键要点\n{points}\n\n"
            f"{_guidance_section(state)}\n\n"
            "注意事项\n"
            "1. 不把部分还款直接说成最终解决，结果以系统入账和政策为准。\n"
            "2. 剩余部分也要形成明确日期或下一次沟通节点。\n"
            "3. 不承诺减免、撤案、停催等超权限结果。\n\n"
            f"{sources}"
        )

    if question_type == "funding_followup":
        return (
            f"结论\n{prefix}核资或还款计划沟通要先尊重客户回应；客户拒绝时不应无限追问，封闭式问题得到答案后应补一次开放式追问以获得有效信息。\n\n"
            f"关键要点\n{points}\n\n"
            f"{_guidance_section(state)}\n\n"
            "注意事项\n"
            "1. 客户明确拒绝不同问题两次后，可以进入下一环节或收口记录。\n"
            "2. 有现实困难时，重点确认困难原因、可处理时间和资金来源，不做主观评价。\n"
            "3. 沟通目标是形成可执行计划，而不是靠持续追问施压。\n\n"
            f"{sources}"
        )

    if question_type == "zero_tolerance":
        return (
            f"结论\n{prefix}零容忍项是严重违规边界，触发后通常不再按普通评分处理，应重点避免不当话术、信息泄露、强催三方、暴力催收、私自催收和投诉倾向未处理等问题。\n\n"
            f"关键要点\n{points}\n\n"
            f"{_guidance_section(state)}\n\n"
            "注意事项\n"
            "1. 不要把零容忍解释成普通扣分项，它代表高风险违规边界。\n"
            "2. 涉及辱骂、威胁、冒充身份、泄露信息、强催联系人等场景应立即保守处理并升级。\n"
            "3. 回答时应解释业务含义，不输出表格列名或评分字段。\n\n"
            f"{sources}"
        )

    if question_type == "third_party_identity_check":
        return (
            f"结论\n{prefix}“核对三方号码、何时办理、何地办理、是否停机”等内容只适用于联系人或三方表示不认识客户、需要核实号码和关系的场景，不应当成普通客户沟通技巧泛化输出。\n\n"
            f"关键要点\n{points}\n\n"
            f"{_guidance_section(state)}\n\n"
            "注意事项\n"
            "1. 该类问题服务于三方身份和关系核验，不用于向三方披露借款人敏感信息。\n"
            "2. 对方无法报姓名时，可核实与欠款人的关系；对方明确否认认识时，应谨慎记录并停止扩大沟通。\n"
            "3. 不要把三方核验话术混入本人阶段沟通或法务表达。\n\n"
            f"{sources}"
        )

    if question_type in {"legal_compliance", "asset_inquiry", "false_commitment"}:
        return (
            f"风险提示\n{prefix}该问题属于合规高风险场景，不能超出知识库证据作出威胁、诉讼进度、冻结后果或减免承诺。\n\n"
            f"可用说法\n{points}\n\n"
            f"{_guidance_section(state)}\n\n"
            "不建议说法\n"
            "1. 不要说“法院一定会冻结银行卡、支付宝、微信”等未经司法程序确认的后果。\n"
            "2. 不要说“下午 5 点正式移交律所/进入诉讼程序”这类无法由当前证据证明的确定性进度。\n"
            "3. 不要承诺撤案、停止联系、减免、消除记录等超出权限的事项。\n\n"
            "需要确认事项\n"
            "1. 请在安全环境中核对当前政策、客户阶段和系统记录。\n"
            "2. 涉及法务表达、资产信息、联系人信息时，建议主管或法务复核。\n\n"
            f"{sources}"
        )

    if question_type == "contact_boundary":
        return (
            f"结论\n{prefix}联系人沟通要以授权、必要性和最小披露为边界；联系人明确不愿被联系或涉及新手机号、家人信息时，不应直接扩大联系。\n\n"
            f"关键依据\n{points}\n\n"
            f"{_guidance_section(state)}\n\n"
            "处理建议\n"
            "1. 只做身份和关系核验，不主动透露借款金额、逾期天数、催收压力、诉讼威胁等敏感信息。\n"
            "2. 联系人明确拒绝被联系时，记录诉求并按规则提交核对，不要继续高频拨打。\n"
            "3. 联系人提供新手机号时，先按内部授权和核验流程处理，不要在未确认授权的情况下直接外呼或传播。\n\n"
            f"{sources}"
        )

    if question_type == "customer_abuse":
        return (
            f"结论\n{prefix}客户骂人时先稳住沟通，不回怼、不刺激；能继续沟通就回到问题和方案，无法沟通时按规则升级或结束通话。\n\n"
            f"处理步骤\n{points}\n\n"
            f"{_guidance_section(state)}\n\n"
            "新人提醒\n"
            "1. 先安抚情绪，避免争辩对错。\n"
            "2. 不使用威胁、讽刺、激将、辱骂回应。\n"
            "3. 记录客户情绪和关键诉求，必要时交由主管复核。\n\n"
            f"{sources}"
        )

    if question_type == "complaint_threat":
        return (
            f"结论\n{prefix}客户提出“骚扰/投诉”时，要先安抚并核对触发点，避免继续刺激客户；涉及频次、联系人、敏感词时应升级复核。\n\n"
            f"处理步骤\n{points}\n\n"
            f"{_guidance_section(state)}\n\n"
            "可执行建议\n"
            "1. 先表达已听到客户反馈，再核对客户反感的是频次、联系人、语气还是具体话术。\n"
            "2. 不反驳“你不能投诉”，也不承诺立即永久停止联系。\n"
            "3. 将投诉风险、客户诉求和后续跟进建议记录到系统。\n\n"
            f"{sources}"
        )

    if question_type == "stage_script":
        return (
            f"结论\n{prefix}阶段话术要跟当前逾期阶段匹配；D4-D6 偏核实和温和提醒，D10-D15 可加强提醒，但不能升级成威胁或不实承诺。\n\n"
            f"阶段要点\n{points}\n\n"
            f"{_guidance_section(state)}\n\n"
            "注意事项\n"
            "1. 强话术只能体现提醒强度和后果提示，不能编造法务进度。\n"
            "2. 每次沟通仍需先确认身份、诉求和当前系统记录。\n"
            "3. 如果客户阶段或逾期天数变化，以当前记录为准。\n\n"
            f"{sources}"
        )

    if question_type == "new_collector":
        return (
            f"结论\n{prefix}新人优先做到核身准确、语气稳定、边界清楚、记录完整，不要急着承诺结果或套用强硬话术。\n\n"
            f"关键要点\n{points}\n\n"
            f"{_guidance_section(state)}\n\n"
            "操作建议\n"
            "1. 开场先自报身份并核实对象。\n"
            "2. 先听客户诉求，再给出符合规则的下一步。\n"
            "3. 遇到投诉、联系人、法务、减免、停止联系等场景及时升级。\n\n"
            f"{sources}"
        )

    if question_type == "negotiation_stage":
        return (
            f"结论\n{prefix}谈判点应以当前逾期天数、当前阶段和系统记录为准，不能继续使用已经不匹配的历史最高逾期天数制造压力。\n\n"
            f"依据\n{points}\n\n"
            f"{_guidance_section(state)}\n\n"
            "建议\n"
            "1. 先核对当前逾期天数和适用话术层级。\n"
            "2. 历史信息只能作为内部分析参考，不应用来误导客户。\n"
            "3. 阶段变化时同步调整提醒强度和沟通目标。\n\n"
            f"{sources}"
        )

    if question_type in {"clarification_medical", "clarification_overseas", "reduction_request"}:
        return (
            f"结论\n{prefix}根据当前知识库暂无法直接确认最终处理口径，需要补充场景并核对政策后再答复。\n\n"
            f"可参考依据\n{points}\n\n"
            f"{_guidance_section(state)}\n\n"
            "澄清/核对项\n"
            "1. 请在安全环境中核对客户当前阶段、系统记录和适用政策。\n"
            "2. 医疗、境外、减免类问题不要现场承诺结果，可先表达理解并说明会按流程核实。\n"
            "3. 如涉及联系人或家人，先确认授权和联系边界。\n\n"
            f"{sources}"
        )

    if question_type == "operation":
        return (
            f"推荐处理方式\n{prefix}先确认场景，再按知识库中的流程执行。\n\n"
            f"操作步骤\n{points}\n\n"
            f"{_guidance_section(state)}\n\n"
            "禁止事项\n1. 不要脱离知识库依据扩大解释。\n2. 不要对客户作出超权限承诺。\n\n"
            f"{sources}"
        )

    if question_type == "knowledge_qa":
        return (
            f"结论\n{prefix}根据当前知识库证据，可以按下列要点理解和执行。\n\n"
            f"关键要点\n{points}\n\n"
            "建议\n"
            "1. 优先按知识库中明确出现的规则、时间、流程或话术执行。\n"
            "2. 如果实际场景与证据不完全一致，请补充材料或提交人工核对。\n\n"
            f"{sources}"
        )

    return (
        f"结论\n{prefix}与客户沟通的核心是：身份清楚、语气稳定、先回应客户诉求，再给出符合规则的下一步动作。\n\n"
        f"关键要点\n{points}\n\n"
        f"{_guidance_section(state)}\n\n"
        "操作建议\n"
        "1. 先确认客户身份、问题背景和当前阶段。\n"
        "2. 客户有情绪时先安抚，再解释规则和可选方案。\n"
        "3. 结束前明确下一步动作、时间点和需要客户配合的事项。\n\n"
        "注意事项\n"
        "1. 不要直接堆叠话术，要根据客户阶段和情绪调整表达。\n"
        "2. 涉及金额、时间、分期、撤案、停止联系等内容时，必须以当前政策和系统记录为准。\n"
        "3. 联系人沟通要注意边界，避免泄露不必要信息或造成投诉风险。\n\n"
        f"{sources}"
    )


def _bullet_points(state: RagWorkflowState, limit: int) -> str:
    points: list[str] = []
    for hit in state.final_hits:
        for sentence in _split_sentences(hit.chunk.text):
            point = _clean_answer_point(sentence)
            if point and point not in points:
                points.append(point)
            if len(points) >= limit:
                break
        if len(points) >= limit:
            break
    if not points and state.context:
        points = [_clean_answer_point(line) for line in state.context.splitlines() if line.startswith("- ")][:limit]
    points = [point for point in points if point]
    points = _complete_points(state, points, limit)
    return "\n".join(f"{index}. {point}" for index, point in enumerate(points[:limit], start=1))


def _low_confidence_answer(state: RagWorkflowState) -> str:
    if state.needs_clarification:
        return (
            "当前知识库没有找到足够明确的依据，暂不建议直接回答。\n\n"
            "可以补充具体场景，例如客户所处阶段、是否涉及法务/投诉/联系人、希望使用的话术类型。"
        )
    return "当前知识库没有足够依据回答该问题。建议补充相关文档后再查询。"


def _guidance_section(state: RagWorkflowState) -> str:
    points = _guidance_points(state)
    return "操作拆解\n" + "\n".join(f"{index}. {point}" for index, point in enumerate(points, start=1))


def _guidance_points(state: RagWorkflowState) -> list[str]:
    question = state.question
    question_type = state.question_type

    if question_type == "customer_abuse":
        return [
            "先让客户把情绪表达完，不抢话、不回怼、不评价对错。",
            "识别客户真实诉求，是频次、金额、态度还是处理方案不满。",
            "能沟通时拉回当前规则和可执行动作；不能沟通时记录并升级。",
        ]

    if question_type == "complaint_threat":
        return [
            "先确认客户认为被骚扰或准备投诉的触发点，例如频次、联系人、语气或具体表述。",
            "当场停止争辩，不说“你不能投诉”，也不承诺永久停止联系。",
            "把投诉风险、客户诉求和后续跟进建议记录到系统，并按规则复核。",
        ]

    if question_type == "stage_script":
        if "D10" in question.upper() or "D15" in question.upper():
            return [
                "先确认当前逾期阶段和多轮跟进记录，避免直接跳到法务或威胁表达。",
                "把沟通目标推进到具体处理安排、具体日期、可处理金额或持续跟进动作。",
                "提醒强度可以更明确，但不能虚构冻结、移交律所、诉讼时间等结果。",
            ]
        if "D4" in question.upper() or "D6" in question.upper():
            return [
                "先完成身份和来意说明，再确认客户是否方便沟通。",
                "重点了解客户困难、还款意愿和可处理时间，提醒强度保持温和。",
                "客户有情绪时先承接情绪，再回到当前记录和下一步安排。",
            ]
        return [
            "先匹配当前逾期阶段和系统记录。",
            "再选择对应强度的沟通目标和提醒边界。",
            "最后确认客户可执行的下一步动作。",
        ]

    if question_type == "new_collector":
        return [
            "开场先说明身份和来意，并完成必要核身。",
            "先听客户诉求，再解释当前规则和可选动作。",
            "遇到投诉、法务、联系人、减免或停催诉求时及时升级。",
        ]

    if question_type == "recognition_encouragement":
        return [
            "先区分“认可沟通意愿”和“认可跳票行为”，只认可前者。",
            "认可后马上回到新的具体还款日期、金额和资金来源。",
            "客户再次跳票时记录原因，并按当前阶段调整跟进强度。",
        ]

    if question_type == "repayment_appointment":
        return [
            "把模糊时间改成今天、明天或具体几月几日。",
            "同步确认还款金额、还款方式和客户需要配合的事项。",
            "如前面处理了异议，收口前再次确认还款时间。",
        ]

    if question_type == "call_protocol":
        return [
            "接通后尽早表明机构身份和来意，再进入核身或沟通目标。",
            "当天非首通也要再次自报家门，避免客户无法识别来电主体。",
            "结束时用礼貌结束语收口，优先使用“再见”。",
        ]

    if question_type == "effective_comfort":
        return [
            "先承接客户情绪或不满点，而不是只要求对方文明用语。",
            "再说明希望把问题说清楚，并引导回当前事项。",
            "持续辱骂无法沟通时，记录情况并按规则升级或结束。",
        ]

    if question_type == "partial_repayment":
        return [
            "确认本次能处理的具体金额和具体日期。",
            "继续确认剩余部分的处理安排和下一次跟进节点。",
            "提醒结果以系统入账和政策记录为准，不作超权限承诺。",
        ]

    if question_type == "funding_followup":
        return [
            "先围绕工作、收入、发薪日、资金来源和现实困难获取有效信息。",
            "封闭式问题得到答案后，补一个开放式问题确认原因或安排。",
            "客户对不同问题明确拒绝两次后，停止持续追问并进入收口记录。",
        ]

    if question_type == "zero_tolerance":
        return [
            "先判断是否涉及辱骂威胁、信息泄露、强催三方、冒充身份或私自催收。",
            "一旦触及零容忍边界，停止普通推进，按投诉或合规风险处理。",
            "对用户解释业务含义，不输出原始评分表头或内部字段。",
        ]

    if question_type == "third_party_identity_check":
        return [
            "先确认这是三方或联系人核验场景，不用于本人沟通。",
            "核实号码办理、使用情况或关系时，只为确认是否认识客户。",
            "不向三方透露借款金额、逾期天数、资产或法务压力。",
        ]

    if question_type == "contact_boundary":
        if "手机号" in question:
            return [
                "先记录号码来源和提供人关系，不现场确认号码真实性。",
                "按授权和核验流程判断是否可以使用该号码。",
                "未确认授权前，不直接外呼、传播或向联系人反馈客户业务细节。",
            ]
        if "停止" in question or "不想" in question:
            return [
                "先记录联系人不愿被联系的诉求和原因。",
                "不要继续高频拨打，也不要现场承诺永久停止联系。",
                "按授权、必要性和系统记录提交核对。",
            ]
        return [
            "只做身份、关系和必要转达核实。",
            "不披露借款金额、逾期天数、催收压力或诉讼威胁。",
            "联系人不方便时记录反馈并按流程处理。",
        ]

    if question_type in {"legal_compliance", "asset_inquiry", "false_commitment"}:
        return [
            "先判断是否涉及法务后果、资产信息、减免承诺或停止联系等高风险内容。",
            "只陈述系统可核实事实和公司流程，不把可能结果说成确定结果。",
            "超出权限的事项记录诉求并提交复核，不现场承诺。",
        ]

    if question_type == "negotiation_stage":
        return [
            "先核对当前逾期天数、当前阶段和系统记录。",
            "历史最高逾期天数只能作为内部分析参考，不能替代当前阶段话术。",
            "阶段变化后同步调整提醒强度和谈判目标。",
        ]

    if question_type == "clarification_medical":
        return [
            "先表达理解并记录客户就医安排，不否定客户困难。",
            "只确认后续可联系时间或可处理节点，不要求客户现场承诺结果。",
            "涉及延期、减免或停催时按政策核对后答复。",
        ]

    if question_type == "clarification_overseas":
        return [
            "先记录客户境外状态和可联系渠道。",
            "是否联系家人要先核对授权、必要性和联系人边界。",
            "未确认前不扩大联系，也不向家人披露客户敏感信息。",
        ]

    if question_type == "reduction_request":
        return [
            "先确认客户减免诉求、原因和当前还款能力。",
            "不现场承诺可减免、减免比例或最终结果。",
            "按当前政策、权限和系统记录提交核对。",
        ]

    if question_type == "operation":
        return [
            "先确认业务场景、客户身份、当前阶段和系统记录。",
            "再按知识库流程拆成可执行步骤。",
            "涉及审批或合规风险时记录并提交复核。",
        ]

    return [
        "先确认客户身份、问题背景和当前阶段。",
        "客户有情绪时先承接情绪，再解释规则和可选方案。",
        "结束前明确下一步动作、时间点和需要客户配合的事项。",
    ]


def _sanitize_final_answer(answer: str, state: RagWorkflowState) -> str:
    blocked_markers = ("chunk_id", "相似度", "鐩镐技搴?", "embedding", "数据库", "鏁版嵁搴?", "debug")
    lines = []
    for line in answer.splitlines():
        if any(marker.lower() in line.lower() for marker in blocked_markers):
            continue
        if _looks_mojibake(line):
            continue
        lines.append(line.rstrip())
    cleaned = "\n".join(line for line in lines if line.strip()).strip()
    if "引用来源" not in cleaned and "鏉ユ簮锛?" not in cleaned and "寮曠敤鏉ユ簮锛?" not in cleaned:
        cleaned = f"{cleaned}\n\n{build_source_text(state.sources)}"
    return cleaned


def _citation_from_state_hit(hit) -> Citation:
    return Citation(
        document_id=hit.chunk.document_id,
        title=str(hit.chunk.metadata.get("doc_name") or hit.chunk.title),
        chunk_id=hit.chunk.chunk_id,
        score=round(hit.score, 4),
        heading_path=list(hit.chunk.metadata.get("heading_path") or []),
        chunk_strategy=str(hit.chunk.metadata.get("chunk_strategy", "")),
    )


def _looks_mojibake(text: str) -> bool:
    markers = (
        "閿?",
        "锟?",
        "閻?",
        "缁旂姾",
        "瀵洜",
        "閸忋儱",
        "闁?",
        "鐎电",
        "Root Entry",
        "SummaryInformation",
        "DocumentSummaryInformation",
        "WordDocument",
        "\ufffd",
    )
    if any(marker in text for marker in markers):
        return True
    suspicious = sum(1 for char in text if ord(char) < 32 and char not in "\t\n\r")
    return suspicious / max(len(text), 1) > 0.03


def _split_sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"[\n。；;.!?？锛焆]+", text) if item.strip()]


def _clean_answer_point(text: str) -> str:
    point = re.sub(r"\s+", " ", text or "").strip(" -\t|")
    point = re.sub(r"^(?:\d+[、.])+", "", point).strip()
    noise_prefixes = ("业务分类", "字段", "column_", "Sheet", "序号", "编号", "DO-", "do-", "参考话术")
    if (
        any(point.startswith(prefix) for prefix in noise_prefixes)
        or _looks_mojibake(point)
        or _looks_like_table_noise(point)
        or _looks_like_generic_call_flow(point)
    ):
        return ""
    return point if len(point) >= 4 else ""


def _looks_like_table_noise(text: str) -> bool:
    lowered = text.lower()
    if "column_" in lowered:
        return True
    if "|" in text and (text.count("|") >= 2 or re.search(r"\|\s*\|", text)):
        return True
    if text.count("为本人/联系人") >= 2 or text.count("本人/联系人") >= 3:
        return True
    if "零容忍" in text and len(text) < 30:
        return True
    if "涉及任意一项" in text and "0" in text and "分" in text:
        return True
    if text.count(",") >= 4 and ("column" in lowered or "本人" in text):
        return True
    return False


def _looks_like_generic_call_flow(text: str) -> bool:
    markers = (
        "委外",
        "受宜享花",
        "结束语",
        "拜拜",
        "核身",
        "核实姓名",
        "参考话术",
        "张三",
        "李四",
        "先生/女士",
        "家人朋友",
        "请问您是",
        "请问是",
        "核对三方号码",
        "何时办理",
        "何地办理",
        "使用期间是否停机",
        "用户画像",
        "年龄\\户籍",
        "当天非首通电话",
        "自报家门",
        "主动挂机",
        "Don't-0分",
    )
    return any(marker in text for marker in markers)


def _complete_points(state: RagWorkflowState, points: list[str], limit: int) -> list[str]:
    question_type = state.question_type
    fallback_by_type = {
        "communication_script": [
            "开场先说明身份和来意，并完成必要的身份核实。",
            "先听客户诉求和情绪，再用稳定语气解释规则和可选方案，避免不实承诺。",
            "给出下一步动作时要具体到事项、时间点和客户需要配合的内容。",
            "涉及金额、减免、停止联系、法务后果等内容时，以当前政策和系统记录为准。",
            "联系人沟通只做必要核实，避免披露借款人敏感信息。",
        ],
        "new_collector": [
            "开场先自报身份并核实对象，不要跳过核身。",
            "优先使用稳定、清楚、可记录的表达，不要急着加强语气。",
            "遇到投诉、联系人、法务、减免、停止联系等场景及时升级。",
        ],
        "customer_abuse": [
            "先安抚情绪，不回怼、不讽刺、不刺激客户。",
            "把话题拉回客户诉求、当前规则和可执行方案。",
            "无法继续沟通时记录情况，并按规则升级或结束通话。",
        ],
        "stage_script": _stage_fallback_points(state.question),
        "complaint_threat": [
            "先确认客户投诉或骚扰感受的触发点。",
            "避免争辩和刺激性表达，不承诺超权限事项。",
            "记录投诉风险并提交主管或合规复核。",
        ],
        "negotiation_stage": [
            "沟通口径应以当前逾期天数、当前阶段和系统记录为准。",
            "历史最高逾期天数只能作为内部分析参考，不应用来误导客户或制造压力。",
            "阶段变化时同步调整提醒强度和沟通目标。",
        ],
        "recognition_encouragement": [
            "客户多次跳票时仍需要认可鼓励，但只能认可其愿意接听、愿意沟通或仍有处理意愿。",
            "不要认可跳票行为本身，也不要用空泛夸奖替代还款计划确认。",
            "认可后应回到具体还款日期、金额、资金来源和后续跟进安排。",
        ],
        "repayment_appointment": [
            "约定还款时间必须明确到今天、明天或具体日期。",
            "不要使用“过两天”“过几天”等模糊时间。",
            "同时确认还款金额、还款方式和后续跟进节点。",
        ],
        "call_protocol": [
            "开场应表明机构身份和来意，并完成必要核身。",
            "当天非首通电话仍需要再次自报家门。",
            "结束时应礼貌收口并说“再见”。",
        ],
        "effective_comfort": [
            "客户骂人或情绪升高时，应先承接情绪，不回怼、不刺激。",
            "单独说“请使用文明用语”不属于有效安抚。",
            "无法继续沟通时记录情况，并按规则升级或结束。",
        ],
        "partial_repayment": [
            "先确认本次可处理的具体金额和具体日期。",
            "再确认剩余部分的处理安排和后续跟进节点。",
            "不要把部分还款承诺成最终解决结果。",
        ],
        "funding_followup": [
            "核资应围绕工作、收入、发薪日、资金来源和现实困难获取有效信息。",
            "封闭式问题得到答案后，应补一次开放式追问。",
            "客户对不同问题明确拒绝两次后，不应无限追问。",
        ],
        "zero_tolerance": [
            "零容忍项是严重违规边界，不是普通扣分项。",
            "重点包括不当话术、信息泄露、强催三方、投诉倾向未处理、信息安全、私自催收和暴力催收等风险。",
            "回答时应解释业务含义，不输出表格列名或内部评分字段。",
        ],
        "third_party_identity_check": [
            "三方号码核对适用于联系人或三方表示不认识客户时的身份和关系核验。",
            "可围绕号码办理、使用情况或关系做必要核实。",
            "不得借核验向三方披露客户借款、逾期、资产或法务信息。",
        ],
    }
    fallback = fallback_by_type.get(question_type, ["根据当前知识库证据，先保持保守表达并补充人工核对。"])
    if question_type in {
        "complaint_threat",
        "negotiation_stage",
        "recognition_encouragement",
        "repayment_appointment",
        "call_protocol",
        "effective_comfort",
        "partial_repayment",
        "funding_followup",
        "zero_tolerance",
        "third_party_identity_check",
    }:
        merged = list(fallback)
        for item in points:
            if len(merged) >= limit:
                break
            if (
                item not in merged
                and not _looks_mojibake(item)
                and not _looks_like_generic_call_flow(item)
                and not _looks_like_table_noise(item)
            ):
                merged.append(item)
        return merged[:limit]
    if question_type == "communication_script":
        merged = list(fallback[:3])
        for item in points:
            if len(merged) >= limit:
                break
            if item not in merged:
                merged.append(item)
        for item in fallback[3:]:
            if len(merged) >= limit:
                break
            if item not in merged:
                merged.append(item)
        return merged
    if question_type == "stage_script":
        merged = list(fallback)
        for item in points:
            if len(merged) >= limit:
                break
            if _is_stage_specific_point(item, state.question) and item not in merged:
                merged.append(item)
        return merged[:limit]
    for item in fallback:
        if len(points) >= limit:
            break
        if item not in points:
            points.append(item)
    return points


def _stage_fallback_points(question: str) -> list[str]:
    if "D10" in question.upper() or "D15" in question.upper():
        return [
            "D10-D15 属于后期跟进阶段，可从普通提醒转为确认可执行处理安排。",
            "可以围绕违约确认、还款义务、处理期限和后续持续跟进做更明确提示。",
            "谈判点应有依据，例如当前逾期记录、客户多轮未处理、征信或合同后果提示。",
            "强度只能体现在提醒更明确、时间点更清楚、跟进动作更具体，不能虚构法务进度。",
            "涉及法律后果、资产信息或家属知情等表达时，应以知识库和合规审核口径为准。",
        ]
    if "D4" in question.upper() or "D6" in question.upper():
        return [
            "D4-D6 属于早期阶段，应以身份核实、情况了解和温和提醒为主。",
            "沟通重点是确认客户当前困难、还款意愿和可处理时间。",
            "不建议过早使用法务进度、移交、资产摸底等加重表达。",
        ]
    return [
        "阶段话术应先匹配当前逾期阶段和系统记录。",
        "沟通强度应随阶段递进，但不能脱离知识库和合规边界。",
        "涉及承诺、法务、联系人或资产信息时需要谨慎核对。",
    ]


def _is_stage_specific_point(point: str, question: str) -> bool:
    if _looks_like_generic_call_flow(point) or _looks_like_table_noise(point):
        return False
    keywords = (
        "D10",
        "D15",
        "D4",
        "D6",
        "阶段",
        "逾期",
        "违约",
        "还款",
        "处理期限",
        "处理安排",
        "谈判",
        "征信",
        "合同",
        "法律风险",
        "法律后果",
        "持续跟进",
        "多次",
        "记录",
        "流程",
    )
    if any(keyword in point for keyword in keywords):
        return True
    return ("D10" in question.upper() or "D15" in question.upper()) and any(
        keyword in point for keyword in ("可使用", "后续", "知悉", "义务", "风险")
    )


def _compact(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def _strategy_counts(chunks) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        strategy = str(chunk.metadata.get("chunk_strategy", "unknown"))
        counts[strategy] = counts.get(strategy, 0) + 1
    return counts


def _ingested_strategy_counts(chunks) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        strategy = str(chunk.metadata.get("chunk_strategy", "unknown"))
        counts[strategy] = counts.get(strategy, 0) + 1
    return counts


def answer_to_dict(answer: RagAnswer) -> dict[str, object]:
    payload = asdict(answer)
    payload["citations"] = [
        {
            "title": citation.title,
            "heading_path": citation.heading_path,
        }
        for citation in answer.citations
    ]
    return payload
