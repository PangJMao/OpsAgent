from __future__ import annotations

import re

from ops_agent.context.context_schema import ConversationState


STAGE_RE = re.compile(r"\bD\s*(\d{1,2})(?:\s*[-~到至]\s*D?\s*(\d{1,2}))?", re.IGNORECASE)


def detect_stage(text: str) -> str:
    normalized = text.upper().replace(" ", "")
    if "D4-D9" in normalized or "D4到D9" in normalized or "D4至D9" in normalized:
        return "D4-D9"
    match = STAGE_RE.search(text)
    if not match:
        return ""
    start = int(match.group(1))
    end = int(match.group(2) or start)
    low, high = min(start, end), max(start, end)
    if low <= 4 and high >= 9:
        return "D4-D9"
    if 4 <= low <= 6 and high <= 6:
        return "D4-D6"
    if 7 <= low <= 9 and high <= 9:
        return "D7-D9"
    if 10 <= low <= 15 and high <= 15:
        return "D10-D15"
    if 4 <= start <= 6:
        return "D4-D6"
    if 7 <= start <= 9:
        return "D7-D9"
    if 10 <= start <= 15:
        return "D10-D15"
    return ""


def detect_topic(text: str) -> str:
    if any(term in text for term in ("资产", "房产", "车产", "存款", "股票", "基金", "资产摸底")):
        return "资产摸底"
    if any(term in text for term in ("安抚", "文明用语", "骂人", "情绪")):
        return "安抚"
    if any(term in text for term in ("联系人", "紧急联系人", "家人", "亲属")):
        return "联系人"
    if any(term in text for term in ("法务", "法院", "冻结", "诉讼", "律所", "移交")):
        return "法务"
    if any(term in text for term in ("承诺", "减免", "停催", "撤案", "不实承诺")):
        return "承诺"
    if any(term in text for term in ("D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15", "阶段")):
        return "阶段话术"
    if any(term in text for term in ("核资", "收入", "工资", "资金来源", "还款能力")):
        return "核资"
    if any(term in text for term in ("沟通", "话术", "技巧")):
        return "客户沟通"
    return ""


def detect_scene(text: str, topic: str = "") -> str:
    if topic == "资产摸底" or any(term in text for term in ("核资", "还款能力", "收入", "工资", "资金")):
        return "核资"
    if topic == "法务":
        return "法务"
    if topic == "联系人":
        return "联系人沟通"
    if topic in {"安抚", "阶段话术", "客户沟通"} or any(term in text for term in ("客户", "沟通", "话术")):
        return "客户沟通"
    return ""


def has_coreference(text: str) -> bool:
    return any(term in text for term in ("那", "这个阶段", "这种情况", "刚才说的", "上面", "继续", "还可以"))


def resolve_coreference(message: str, state: ConversationState) -> str:
    if not has_coreference(message):
        return message
    parts: list[str] = []
    if state.current_stage and not detect_stage(message):
        parts.append(f"{state.current_stage} 阶段")
    if state.current_topic and not detect_topic(message):
        parts.append(state.current_topic)
    if not parts:
        return message
    clean = message
    if clean.startswith("那"):
        clean = clean[1:].strip()
    prefix = "，".join(parts)
    return f"{prefix}：{clean}"


def update_state_from_message(
    state: ConversationState,
    message: str,
    *,
    intent: str = "",
    decision: str = "",
    sources: list[str] | None = None,
) -> ConversationState:
    stage = detect_stage(message) or state.current_stage
    topic = detect_topic(message) or state.current_topic
    scene = detect_scene(message, topic) or state.current_scene
    return ConversationState(
        current_topic=topic,
        current_stage=stage,
        current_scene=scene,
        last_intent=intent or state.last_intent,
        last_decision=decision or state.last_decision,
        last_sources=sources if sources is not None else list(state.last_sources),
    )
