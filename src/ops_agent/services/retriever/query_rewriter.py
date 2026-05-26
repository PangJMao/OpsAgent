from __future__ import annotations

import re

from ops_agent.config.retrieval_config import DEFAULT_RETRIEVAL_CONFIG, RetrievalConfig


EXACT_TERMS = (
    "D4-D6",
    "D7-D9",
    "D10-D15",
    "黑名单",
    "敏感词",
    "投诉",
    "联系人",
    "紧急联系人",
    "预留联系人",
    "法务",
    "诉讼",
    "冻结",
    "承诺",
    "不实承诺",
    "逾期",
    "房产",
    "车产",
    "存款",
    "股票",
    "基金",
    # Mojibake variants kept for compatibility with existing fixtures.
    "榛戝悕鍗?",
    "鏁忔劅璇?",
    "鎶曡瘔",
    "鑱旂郴浜?",
    "娉曞姟",
    "鎵胯",
    "閫炬湡",
)


class QueryRewriter:
    """Rule-based query rewrite for stable recall of business terms."""

    def __init__(self, config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG) -> None:
        self.config = config

    def rewrite(self, question: str, question_type: str = "knowledge_qa") -> list[str]:
        queries = [question.strip().rstrip("？?")]

        if question_type in {"customer_abuse", "complaint_threat"}:
            queries.extend(
                [
                    "客户辱骂 骂人 情绪激动 安抚 文明用语 禁止回怼",
                    "客户投诉 骚扰 投诉风险 安抚 解释边界 升级处理",
                    "敏感词 投诉 客户不满 如何沟通 禁止威胁 禁止刺激",
                ]
            )

        if question_type == "new_collector":
            queries.extend(
                [
                    "新人 催收沟通 注意事项 核身 自报家门 语气 话术边界",
                    "新员工 客户沟通 合规要求 不实承诺 敏感词 投诉风险",
                    "催收新人 沟通流程 身份核实 客户诉求 下一步动作",
                ]
            )

        if question_type == "stage_script":
            if stage := _stage_from_question(question):
                queries.extend(
                    [
                        f"{stage} 阶段 客户沟通 分阶段 话术 谈判点",
                        f"{stage} 逾期阶段 沟通策略 提醒 强度 合规",
                    ]
                )
            if "D10" in question.upper() or "D15" in question.upper():
                queries.append("D10-D15 更强话术 强提醒 谈判点 禁止威胁 不实承诺")
            if "D4" in question.upper() or "D6" in question.upper():
                queries.append("D4-D6 温和沟通 信息核实 还款意愿 客户安抚")

        if question_type in {"contact_boundary", "privacy_boundary"}:
            queries.extend(
                [
                    "联系人 预留联系人 紧急联系人 沟通边界 不得透露 借款信息",
                    "联系人明确不想被联系 后续拨打 黑名单 停止联系 投诉风险",
                    "联系人提供新手机号 核实手机号 隐私 授权 安全环境",
                ]
            )

        if question_type in {"legal_compliance", "asset_inquiry", "false_commitment"}:
            queries.extend(
                [
                    "法务话术 合规风险 禁止威胁 禁止不实承诺 诉讼 冻结",
                    "不实承诺 风险点 催收沟通 承诺撤案 停止联系 减免",
                    "法院 冻结 银行卡 支付宝 微信 律所 诉讼 禁止绝对化",
                    "房产 车产 存款 股票 基金 资产信息 敏感信息 是否允许",
                ]
            )

        if question_type in {"negotiation_stage", "reduction_request"}:
            queries.extend(
                [
                    "逾期天数下降 历史最高逾期天数 谈判点 当前阶段 不得误导",
                    "减免利息 减免政策 优惠 权限 系统记录 人工核对",
                    "还款方案 谈判点 当前逾期天数 当前政策",
                ]
            )

        if question_type in {"clarification_medical", "clarification_overseas"}:
            queries.extend(
                [
                    "客户去医院 特殊情况 人文关怀 暂缓沟通 后续跟进",
                    "客户在国外 联系家人 授权 联系人边界 隐私合规",
                    "证据不足 需要补充 客户场景 人工核对",
                ]
            )

        if question_type == "communication_script" or any(
            word in question for word in ("客户", "沟通", "话术", "催收", "瀹㈡埛", "娌熼€?", "璇濇湳")
        ):
            queries.extend(
                [
                    "客户沟通技巧 客户不满 安抚 认可鼓励 敏感词 语气 语速 身份说明",
                    "客户沟通 话术 分阶段 D4-D6 D7-D9 D10-D15",
                    "客户沟通 注意事项 不实承诺 联系人沟通 联系人边界 合规",
                    "瀹㈡埛娌熼€氭妧宸?瀹夋姎 璁ゅ彲榧撳姳 璇皵 璇€?",
                    "瀹㈡埛涓嶆弧 鎶曡瘔 鏁忔劅璇?濡備綍娌熼€?鏂囨槑鐢ㄨ",
                    "瀹㈡埛娌熐?璇濇湳 鍒嗛樁娈?D4-D6 D7-D9 D10-D15",
                ]
            )

        if any(word in question for word in ("法务", "诉前", "起诉", "律师函", "冻结", "法院", "律所", "娉曞姟", "璇夊墠", "璧疯瘔")):
            queries.extend(["法务话术 诉前沟通 合规风险", "可用说法 不建议说法 禁止承诺 禁止威胁"])

        if stage := _stage_from_question(question):
            queries.append(f"{stage} 客户沟通 分阶段 话术")

        return _unique([query for query in queries if query])[: self.config.rewritten_query_limit]


def has_exact_business_terms(question: str) -> bool:
    return any(term in question for term in EXACT_TERMS) or bool(re.search(r"D\d+\s*[-~至到鑷冲埌]\s*D?\d+", question, re.I))


def _stage_from_question(question: str) -> str:
    match = re.search(r"D\d+\s*[-~至到鑷冲埌]\s*D?\d+", question, re.I)
    return match.group(0) if match else ""


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result
