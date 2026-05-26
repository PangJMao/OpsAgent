from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BusinessFrame:
    stage: str = ""
    subject: str = "本人"
    action: str = ""
    information: tuple[str, ...] = ()
    risk_tags: tuple[str, ...] = ()
    contact_round: str = ""
    customer_emotion: str = ""
    question_type: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_business_frame(question: str, question_type: str = "") -> BusinessFrame:
    stage = stage_bucket(question)
    subject = subject_bucket(question)
    action = action_bucket(question, question_type)
    information = information_types(question)
    risk_tags = risk_tags_for(question, action, information)
    return BusinessFrame(
        stage=stage,
        subject=subject,
        action=action,
        information=tuple(information),
        risk_tags=tuple(risk_tags),
        contact_round=contact_round_bucket(question),
        customer_emotion=emotion_bucket(question),
        question_type=question_type,
    )


def stage_bucket(question: str) -> str:
    text = question.upper().replace(" ", "")
    if ("还没到D10" in text or "未到D10" in text or "不到D10" in text) and contains_any(
        question, ("资产", "房产", "车产", "存款", "核资")
    ):
        return "D7-D9"

    single = re.search(r"D\s*(\d{1,2})", text, flags=re.IGNORECASE)
    if single:
        day = int(single.group(1))
        if 10 <= day <= 15:
            return "D10-D15"
        if 7 <= day <= 9:
            return "D7-D9"
        if 4 <= day <= 6:
            return "D4-D6"

    if re.search(r"D\s*10\s*[-~到至]\s*D?\s*15", question, flags=re.IGNORECASE):
        return "D10-D15"
    if re.search(r"D\s*4\s*[-~到至]\s*D?\s*9", question, flags=re.IGNORECASE):
        return "D7-D9"
    if re.search(r"D\s*4\s*[-~到至]\s*D?\s*6", question, flags=re.IGNORECASE):
        return "D4-D6"
    return ""


def subject_bucket(question: str) -> str:
    if contains_any(question, ("联系人", "紧急联系人", "预留联系人", "家人", "亲属")):
        return "联系人"
    return "本人"


def contact_round_bucket(question: str) -> str:
    if contains_any(question, ("第一次接通", "首次接通", "第一次通话", "首通", "第一次联系")):
        return "首通"
    if contains_any(question, ("再次接通", "后续接通", "多次沟通", "多轮跟进", "复通", "再次联系")):
        return "复通"
    return ""


def emotion_bucket(question: str) -> str:
    if contains_any(question, ("投诉", "骚扰", "我要投诉")):
        return "投诉"
    if contains_any(question, ("骂人", "辱骂")):
        return "辱骂"
    if contains_any(question, ("有点不满", "有些不满", "有点情绪", "有些情绪", "有点生气")):
        return "轻度不满"
    if contains_any(question, ("不满", "生气", "情绪不好", "情绪较大")):
        return "不满"
    return ""


def action_bucket(question: str, question_type: str) -> str:
    if contains_any(question, ("核资", "还款能力", "资金来源", "收入", "工资", "工作单位", "可调配资金")):
        return "核资"
    if question_type == "asset_inquiry" or contains_any(question, ("房产", "车产", "存款", "股票", "基金", "资产")):
        return "资产摸底"
    if question_type == "legal_compliance" or contains_any(question, ("法院", "冻结", "诉讼", "律所", "移交", "起诉")):
        return "法务表达"
    if question_type == "contact_boundary":
        return "联系人沟通"
    if question_type in {"false_commitment", "reduction_request"}:
        return "承诺"
    if question_type in {"stage_script", "communication_script"} and contains_any(question, ("沟通", "怎么沟通", "如何沟通", "话术")):
        return "阶段沟通"
    return ""


def information_types(question: str) -> list[str]:
    mapping = (
        ("资产", ("房产", "车产", "存款", "股票", "基金", "资产")),
        ("收入", ("收入", "工资", "发薪", "工作", "资金来源")),
        ("联系方式", ("手机号", "电话", "号码")),
        ("法务", ("法院", "冻结", "诉讼", "律所", "起诉")),
        ("减免", ("减免", "利息", "优惠")),
        ("联系人", ("联系人", "家人", "亲属")),
    )
    return [label for label, terms in mapping if contains_any(question, terms)]


def risk_tags_for(question: str, action: str, information: tuple[str, ...] | list[str]) -> list[str]:
    tags: list[str] = []
    if "联系人" in information:
        tags.append("隐私披露")
    if action in {"资产摸底", "核资"} and "资产" in information:
        tags.append("敏感信息")
    if action == "法务表达" or "法务" in information:
        tags.append("虚假法务")
    if action == "承诺":
        tags.append("不实承诺")
    if contains_any(question, ("必须", "马上", "后果", "冻结", "移交")):
        tags.append("施压表达")
    return tags


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)
