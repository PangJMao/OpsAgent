import types

from ops_agent.models import Chunk, RetrievalHit
from ops_agent.services import RagService


def _hit(chunk_id: str, text: str, score: float = 0.95, **metadata) -> RetrievalHit:
    return RetrievalHit(
        Chunk(
            chunk_id=chunk_id,
            document_id="dialog-policy",
            title=metadata.get("doc_name", "催收沟通合规手册"),
            text=text,
            start_char=0,
            end_char=len(text),
            metadata=metadata,
        ),
        score,
    )


class DialogRegressionStore:
    def __init__(self) -> None:
        self.rows = [
            _hit(
                "general",
                "客户沟通时应先自报身份并核实客户身份，语气保持稳定，先回应客户诉求，再说明符合规则的下一步动作。新人应避免直接套用强硬话术，遇到投诉、法务、减免、联系人边界等场景及时升级。",
                doc_name="话术-沟通话术四大框架",
                sheet_name="沟通原则",
                topic="客户沟通技巧",
                business_scene="新人沟通",
                risk_level="low",
            ),
            _hit(
                "abuse",
                "客户辱骂或情绪激动时，催收员应先安抚，不回怼、不讽刺、不刺激客户；若无法继续沟通，应记录情况并按规则升级或结束通话。",
                doc_name="话术-沟通话术四大框架",
                sheet_name="安抚",
                topic="客户辱骂",
                business_scene="客户不满",
                risk_level="medium",
            ),
            _hit(
                "stage-d4",
                "D4-D6 阶段以身份核实、温和提醒、还款意愿确认和客户安抚为主，不应使用威胁或诉讼进度类表达。",
                doc_name="核资话术分层建议",
                sheet_name="阶段话术",
                topic="D4-D6",
                applicable_stage="D4-D6",
                risk_level="medium",
            ),
            _hit(
                "stage-d10",
                "D10-D15 阶段可以适当加强提醒强度和谈判点，但不得使用不实承诺、威胁、虚构法务进度或确定性冻结后果。",
                doc_name="核资话术分层建议",
                sheet_name="阶段话术",
                topic="D10-D15",
                applicable_stage="D10-D15",
                risk_level="high",
            ),
            _hit(
                "contact",
                "与预留联系人或紧急联系人沟通时，应遵守最小披露原则，不得透露借款金额、逾期天数、催收压力、诉讼威胁等借款人敏感信息；联系人明确不愿被联系时，应记录诉求并提交核对，不要继续高频拨打。",
                doc_name="联系人沟通边界",
                sheet_name="联系人",
                topic="联系人边界",
                business_scene="联系人沟通",
                risk_level="high",
            ),
            _hit(
                "phone",
                "联系人提供客户新的手机号时，应先按内部授权和核验流程处理，不得在未确认授权和来源可靠性的情况下直接外呼或传播。",
                doc_name="联系人沟通边界",
                sheet_name="手机号",
                topic="手机号核验",
                business_scene="联系人沟通",
                risk_level="high",
            ),
            _hit(
                "negotiation",
                "客户当前逾期天数下降后，沟通口径应以当前逾期天数、当前阶段和系统记录为准，历史最高逾期天数只能作为内部分析参考，不应用来误导客户或制造压力。",
                doc_name="谈判点使用规范",
                sheet_name="逾期阶段",
                topic="谈判点",
                business_scene="逾期天数变化",
                risk_level="medium",
            ),
            _hit(
                "legal",
                "法务和合规话术不得承诺法院一定冻结银行卡、支付宝、微信，不得虚构下午五点移交律所或正式进入诉讼程序；涉及诉讼、冻结、律所进度，应以真实流程和法务复核为准。",
                doc_name="法务话术",
                sheet_name="合规边界",
                topic="法务合规",
                business_scene="诉讼表达",
                risk_level="high",
            ),
            _hit(
                "promise",
                "不实承诺包括承诺撤案、停止联系、减免利息、消除记录、保证不会再联系等超出权限或未经系统确认的表达。此类内容需要主管或政策核对。",
                doc_name="法务话术",
                sheet_name="不实承诺",
                topic="不实承诺",
                business_scene="合规风险",
                risk_level="high",
            ),
            _hit(
                "clarify",
                "客户涉及医院、境外、减免等特殊场景时，应先表达理解，再说明会按流程核实；不得现场承诺减免、停止联系或联系家人，必要时补充授权、当前阶段和政策依据。",
                doc_name="特殊场景处理",
                sheet_name="澄清",
                topic="澄清核对",
                business_scene="特殊场景",
                risk_level="medium",
            ),
        ]

    def search(self, query: str, top_k: int = 10):
        return self.rows[:top_k]

    def keyword_search(self, query: str, top_k: int = 10):
        tokens = [token for token in _terms() if token in query]
        if not tokens:
            return self.rows[:top_k]
        hits = []
        for row in self.rows:
            haystack = f"{row.chunk.text} {row.chunk.metadata}"
            if any(token in haystack for token in tokens):
                hits.append(RetrievalHit(row.chunk, row.score + len(tokens)))
        return hits[:top_k]


class FakeReranker:
    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int = 8):
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]


def _service() -> RagService:
    return RagService(
        vector_store=DialogRegressionStore(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        llm=types.SimpleNamespace(enabled=False),
    )


def _terms() -> list[str]:
    return [
        "客户",
        "新人",
        "骂人",
        "辱骂",
        "D4-D6",
        "D10-D15",
        "联系人",
        "紧急联系人",
        "预留联系人",
        "逾期天数",
        "谈判点",
        "骚扰",
        "投诉",
        "停止联系",
        "手机号",
        "不实承诺",
        "冻结",
        "法院",
        "律所",
        "诉讼",
        "医院",
        "国外",
        "家人",
        "减免",
        "利息",
    ]


def test_round_1_basic_usability_questions_are_answerable() -> None:
    service = _service()
    expectations = {
        "与客户沟通时有哪些技巧？": ["身份", "语气", "引用来源"],
        "新人刚开始跟客户沟通时需要注意哪些？": ["新人", "核实", "升级"],
        "客户骂人时，催收员应该怎么处理？": ["不回怼", "安抚", "升级"],
        "D4-D6 阶段客户应该怎么沟通？": ["D4-D6", "温和", "威胁"],
        "D10-D15 阶段可以使用哪些更强的话术？": ["D10-D15", "加强", "不实承诺"],
        "联系人明确表示不想被联系，后续还能继续拨打吗？": ["联系人", "不愿被联系", "高频拨打"],
        "客户当前逾期天数下降了，还能用历史最高逾期天数对应的谈判点吗？": ["当前逾期天数", "历史最高逾期", "误导"],
        "客户说“你们骚扰我，我要投诉”，新人应该怎么处理？": ["投诉", "核对", "升级"],
    }
    for question, phrases in expectations.items():
        answer = service.ask(question)

        assert answer.refused is False, question
        assert "操作拆解" in answer.answer, question
        assert "示例话术" not in answer.answer, question
        for phrase in phrases:
            assert phrase in answer.answer, question


def test_round_2_compliance_boundary_questions_do_not_overpromise() -> None:
    service = _service()
    expectations = {
        "客户要求停止联系紧急联系人，催收员可以直接答应吗？": ["不应直接", "记录诉求", "核对"],
        "与预留联系人沟通时，可以提及哪些借款人信息？": ["最小披露", "不得透露", "敏感信息"],
        "联系人提供了客户新的手机号，可以直接核实这个手机号吗？": ["授权", "核验流程", "直接外呼"],
        "什么是不实承诺？举几个催收沟通中的风险点。": ["不实承诺", "超出权限", "减免"],
        "能不能告诉客户法院会冻结他的银行卡、支付宝、微信？": ["不要说", "冻结", "司法程序"],
        "客户不接受调解，能不能直接说“下午 5 点移交律所并正式进入诉讼程序”？": ["不要说", "移交律所", "诉讼程序"],
    }
    for question, phrases in expectations.items():
        answer = service.ask(question)

        assert answer.refused is False, question
        assert "操作拆解" in answer.answer, question
        assert "示例话术" not in answer.answer, question
        assert "风险" in answer.answer or "边界" in answer.answer
        for phrase in phrases:
            assert phrase in answer.answer, question


def test_round_3_clarification_questions_are_conservative() -> None:
    service = _service()
    expectations = {
        "客户说他要去医院，催收员应该怎么说？": ["暂无法", "理解", "核对"],
        "客户在国外，能不能联系他的家人？": ["暂无法", "授权", "家人"],
        "客户问能不能减免利息，应该怎么答？": ["暂无法", "减免", "政策"],
    }
    for question, phrases in expectations.items():
        answer = service.ask(question)

        assert answer.refused is False, question
        assert "操作拆解" in answer.answer, question
        assert "示例话术" not in answer.answer, question
        for phrase in phrases:
            assert phrase in answer.answer, question


def test_table_noise_columns_are_not_rendered_as_business_points() -> None:
    class DirtyStore(DialogRegressionStore):
        def __init__(self) -> None:
            super().__init__()
            self.rows.insert(
                0,
                _hit(
                    "dirty-columns",
                    "(以终为始)为本人/联系人,column_3为本人/联系人,column_4为本人/联系人,column_5为本人/联系人,沟通环境。"
                    "(营造氛围)为本人/联系人,column_7为本人/联系人,沟通技巧。"
                    "(涉及任意一项则该通录音为“0”分)为本人/联系人,column_14为本人,column_15为本人/联系人。",
                    3.0,
                    doc_name="话术-沟通话术四大框架(2)",
                    sheet_name="Sheet3",
                    topic="沟通技巧",
                    business_scene="客户沟通",
                    risk_level="low",
                ),
            )

    service = RagService(
        vector_store=DirtyStore(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        llm=types.SimpleNamespace(enabled=False),
    )

    answer = service.ask("与客户沟通时有哪些技巧？")

    assert answer.refused is False
    assert "column_" not in answer.answer
    assert "为本人/联系人" not in answer.answer
    assert "涉及任意一项" not in answer.answer
    assert "操作拆解" in answer.answer
    assert "示例话术" not in answer.answer
    assert "开场先说明身份" in answer.answer


def test_pipe_table_noise_is_not_rendered_for_stage_script() -> None:
    class DirtyStageStore(DialogRegressionStore):
        def __init__(self) -> None:
            super().__init__()
            self.rows.insert(
                0,
                _hit(
                    "dirty-pipes",
                    "(以终为始) | | | | 沟通环境。(营造氛围) | | 沟通技巧。"
                    "(促成还款) | | | | 解决方案。(达成共识) | *零容忍。"
                    "(涉及任意一项则该通录音为“0”分)。",
                    3.0,
                    doc_name="话术-沟通话术四大框架(2)",
                    sheet_name="Sheet3",
                    topic="D10-D15",
                    applicable_stage="D10-D15",
                    risk_level="high",
                ),
            )

    service = RagService(
        vector_store=DirtyStageStore(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        llm=types.SimpleNamespace(enabled=False),
    )

    answer = service.ask("D10-D15 阶段可以使用哪些更强的话术？")

    assert answer.refused is False
    assert "| |" not in answer.answer
    assert "零容忍" not in answer.answer
    assert "涉及任意一项" not in answer.answer
    assert "操作拆解" in answer.answer
    assert "示例话术" not in answer.answer
    assert "D10-D15" in answer.answer
    assert "不实承诺" in answer.answer


def test_stage_script_does_not_render_opening_or_identity_verification_snippets() -> None:
    class DirtyStageStore(DialogRegressionStore):
        def __init__(self) -> None:
            super().__init__()
            self.rows.insert(
                0,
                _hit(
                    "dirty-identity",
                    "委外:受宜享花(平台)委托。结束时,使用礼貌的结束语”再见“,“拜拜/88”。"
                    "按照核身话术执行(客户需全名),三方核身同样需要核实姓名/姓氏。"
                    "参考话术:请问是张三先生/女士吗？请问您是李四先生/女士的家人朋友吗？",
                    3.0,
                    doc_name="话术-沟通话术四大框架(2)",
                    sheet_name="Sheet3",
                    topic="D10-D15",
                    applicable_stage="D10-D15",
                    risk_level="high",
                ),
            )

    service = RagService(
        vector_store=DirtyStageStore(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        llm=types.SimpleNamespace(enabled=False),
    )

    answer = service.ask("D10-D15 阶段可以使用哪些更强的话术？")

    assert answer.refused is False
    for dirty in ["委外", "拜拜", "核身", "张三", "李四", "家人朋友"]:
        assert dirty not in answer.answer
    for clean in ["违约确认", "还款义务", "处理期限", "不实承诺"]:
        assert clean in answer.answer


def test_guidance_steps_are_adapted_and_do_not_cross_boundaries() -> None:
    service = _service()

    d10 = service.ask("D10-D15 阶段可以使用哪些更强的话术？").answer
    assert "操作拆解" in d10
    assert "处理安排" in d10
    assert "冻结" not in _guidance_only(d10)
    assert "移交律所" not in _guidance_only(d10)
    assert "诉讼程序" not in _guidance_only(d10)

    reduction = service.ask("客户问能不能减免利息，应该怎么答？").answer
    assert "操作拆解" in reduction
    assert "不能现场承诺" in reduction
    assert "保证减免" not in reduction
    assert "可以减免" not in _guidance_only(reduction)

    contact = service.ask("与预留联系人沟通时，可以提及哪些借款人信息？").answer
    assert "操作拆解" in contact
    guidance = _guidance_only(contact)
    assert "不提借款金额" in guidance or "不披露" in guidance or "最小披露" in guidance
    assert "逾期天数" in contact

    legal = service.ask("能不能告诉客户法院会冻结他的银行卡、支付宝、微信？").answer
    assert "操作拆解" in legal
    assert "不能直接承诺" in legal
    assert "法院会冻结" not in _guidance_only(legal)


def test_asset_inquiry_uses_structured_stage_rules_across_question_variants() -> None:
    service = _service()
    questions = [
        "D4-D9 阶段能不能询问客户名下是否有房产、车产、存款、股票、基金？",
        "D7 客户没还款，能不能问他名下车房存款？",
        "还没到 D10，可以做资产摸底吗？",
    ]

    for question in questions:
        answer = service.ask(question).answer

        assert "结论" in answer
        assert "不建议使用" in answer
        assert "资产摸底" in answer
        assert "D4-D9" in answer or "D4-D6、D7-D9" in answer
        assert "房产" in answer
        assert "冻结" not in _guidance_only(answer)
        assert "column_" not in answer
        assert "核对三方号码" not in answer


def test_d10_asset_inquiry_allows_only_compliant_capacity_check() -> None:
    service = _service()

    answer = service.ask("D10-D15 阶段能不能了解客户有没有房产、车产、存款、股票、基金？").answer

    assert "可在合规边界内谨慎使用" in answer
    assert "还款能力" in answer or "资金来源" in answer
    assert "不要说“查到你有资产后法院一定会冻结”" in answer
    assert "不要向联系人询问或透露客户资产信息" in answer


def test_d12_funding_check_uses_rule_layer_instead_of_generic_communication_answer() -> None:
    service = _service()

    answer = service.ask("客户 D12 阶段，多次沟通未还款，可以问哪些核资问题？").answer

    assert "结论" in answer
    assert "核资" in answer
    assert "D10-D15" in answer
    assert "还款能力" in answer
    assert "工资" in answer or "收入" in answer
    assert "可调配资金" in answer
    assert "操作拆解" in answer
    assert "示例话术" not in answer
    assert "多次沟通" in answer
    assert "(以终为始)" not in answer
    assert "当天非首通电话" not in answer
    assert "column_" not in answer


def test_business_rule_layer_handles_new_business_by_frame_and_config() -> None:
    service = _service()

    answer = service.ask("D13 多轮跟进没有结果，想确认客户收入和资金来源，应该怎么问？").answer

    assert "场景识别" in answer
    assert "阶段：D10-D15" in answer
    assert "行为类型：核资" in answer
    assert "收入" in answer
    assert "资金来源" in answer
    assert "不建议说法" in answer


def test_scene_frame_is_shared_with_workflow_debug() -> None:
    service = _service()

    state = service.workflow.run("客户 D12 阶段，多次沟通未还款，可以问哪些核资问题？")

    assert state.business_frame is not None
    assert state.business_frame.stage == "D10-D15"
    assert state.business_frame.action == "核资"
    assert state.business_frame.contact_round == "复通"
    assert state.debug["business_frame"]["stage"] == "D10-D15"


def test_rule_layer_combines_specific_and_general_rules() -> None:
    service = _service()

    answer = service.ask("客户 D5 阶段，第一次接通电话，情绪有点不满，应该怎么沟通？").answer

    assert "适用规则" in answer
    assert "D4-D6 阶段" in answer
    assert "阶段沟通" in answer
    assert "stage-d4-first-contact-dissatisfied" not in answer
    assert "stage-d4-general" not in answer


def test_d5_first_contact_with_mild_dissatisfaction_uses_scene_frame() -> None:
    service = _service()

    answer = service.ask("客户 D5 阶段，第一次接通电话，情绪有点不满，应该怎么沟通？").answer

    assert "场景识别" in answer
    assert "阶段：D4-D6" in answer
    assert "通话轮次：首通" in answer
    assert "客户情绪：轻度不满" in answer
    assert "先接住情绪" in answer
    assert "温和提醒" in answer
    assert "操作拆解" in answer
    assert "示例话术" not in answer
    assert "说明身份和来意" in answer
    assert "先回应客户的不满情绪" in answer
    assert "还款意愿和可处理时间" in answer
    assert "不要在 D5 首通场景使用法务进度" in answer
    assert "重点跟进" not in _guidance_only(answer)


def test_quality_specific_questions_do_not_fall_back_to_generic_stage_summary() -> None:
    service = _service()

    cases = {
        "客户多次跳票，还需要认可鼓励吗？": ["仍然需要", "不能认可", "跳票"],
        "与客户约定还款时间时，可以说“过两天还”吗？": ["不能", "具体日期", "模糊"],
        "当天不是首通电话，还需要再次自报家门吗？": ["需要", "再次自报家门", "非首通"],
        "客户骂人时，说“请你文明用语”算有效安抚吗？": ["不算", "有效安抚", "承接情绪"],
    }

    for question, phrases in cases.items():
        answer = service.ask(question).answer

        assert "根据当前知识库证据，可以按下列要点理解和执行" not in answer
        assert "操作拆解" in answer
        for phrase in phrases:
            assert phrase in answer


def test_quality_boundary_questions_filter_binary_and_table_noise() -> None:
    class DirtyStore(DialogRegressionStore):
        def __init__(self) -> None:
            super().__init__()
            self.rows.insert(
                0,
                _hit(
                    "binary-doc",
                    "Root Entry SummaryInformation DocumentSummaryInformation WordDocument \ufffd \ufffd \ufffd",
                    5.0,
                    doc_name="法务话术",
                    topic="法务",
                    risk_level="high",
                ),
            )

    service = RagService(
        vector_store=DirtyStore(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        llm=types.SimpleNamespace(enabled=False),
    )

    legal = service.ask("能不能说“你后续一切法律责任都要自己承担”？").answer
    zero = service.ask("什么是“零容忍”项？系统应该怎么解释？").answer
    third_party = service.ask("“核对三方号码、何时办理、何地办理、是否停机”这些内容应该在什么场景使用？").answer

    for answer in [legal, zero, third_party]:
        assert "Root Entry" not in answer
        assert "WordDocument" not in answer
        assert "column_" not in answer
        assert "根据当前知识库证据，可以按下列要点理解和执行" not in answer
    assert "法律责任" in legal
    assert "零容忍" in zero
    assert "三方" in third_party


def _guidance_only(answer: str) -> str:
    if "操作拆解" not in answer:
        return ""
    section = answer.split("操作拆解", 1)[1]
    for marker in ("注意事项", "处理建议", "不建议说法", "需要确认事项", "引用来源", "澄清/核对项", "建议"):
        if marker in section:
            section = section.split(marker, 1)[0]
    return section
