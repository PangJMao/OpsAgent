from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ops_agent.services.business_scene import BusinessFrame, parse_business_frame
from ops_agent.services.rag_workflow import RagWorkflowState, build_source_text


RULE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "resources" / "business_rules.json"


@dataclass(frozen=True)
class BusinessRule:
    rule_id: str
    question_types: tuple[str, ...]
    stages: tuple[str, ...]
    actions: tuple[str, ...]
    subjects: tuple[str, ...]
    contact_rounds: tuple[str, ...]
    emotions: tuple[str, ...]
    decision: str
    rationale: tuple[str, ...]
    allowed_items: tuple[str, ...]
    blocked_scripts: tuple[str, ...]
    sources: tuple[str, ...]
    priority: int = 50


@lru_cache(maxsize=1)
def load_business_rules() -> tuple[BusinessRule, ...]:
    payload = json.loads(RULE_CONFIG_PATH.read_text(encoding="utf-8"))
    return tuple(_rule_from_dict(item) for item in payload.get("rules", []))


def compose_business_rule_answer(state: RagWorkflowState) -> str | None:
    frame = parse_business_frame(state.question, state.question_type)
    if state.question_type not in _rule_types() and not frame.action:
        return None

    rules = select_rules(state.question_type, frame)
    if not rules:
        return None

    return _render_rule_answer(state, frame, rules)


def select_rule(question_type: str, frame: BusinessFrame) -> BusinessRule | None:
    rules = select_rules(question_type, frame)
    return rules[0] if rules else None


def select_rules(question_type: str, frame: BusinessFrame, limit: int = 3) -> list[BusinessRule]:
    candidates: list[tuple[int, BusinessRule]] = []
    for rule in load_business_rules():
        score = _rule_score(rule, question_type, frame)
        if score <= 0:
            continue
        candidates.append((score, rule))

    if not candidates:
        return []

    ordered = [rule for _, rule in sorted(candidates, key=lambda item: item[0], reverse=True)]
    selected: list[BusinessRule] = []
    for rule in ordered:
        if not selected:
            selected.append(rule)
            continue
        if _is_complementary_rule(rule, selected, frame):
            selected.append(rule)
        if len(selected) >= limit:
            break
    return selected


def _rule_score(rule: BusinessRule, question_type: str, frame: BusinessFrame) -> int:
    if question_type not in rule.question_types and frame.action not in rule.actions:
        return 0

    score = rule.priority
    if _matches_dimension(frame.stage, rule.stages):
        score += 30
    elif _requires_specific_value(rule.stages):
        return 0

    if _matches_dimension(frame.action, rule.actions):
        score += 20
    elif _requires_specific_value(rule.actions):
        return 0

    if _matches_dimension(frame.subject, rule.subjects):
        score += 10
    elif _requires_specific_value(rule.subjects):
        return 0

    if _matches_dimension(frame.contact_round, rule.contact_rounds):
        score += 8
    elif _requires_specific_value(rule.contact_rounds):
        return 0

    if _matches_dimension(frame.customer_emotion, rule.emotions):
        score += 8
    elif _requires_specific_value(rule.emotions):
        return 0

    return score


def _is_complementary_rule(rule: BusinessRule, selected: list[BusinessRule], frame: BusinessFrame) -> bool:
    if any(rule.rule_id == item.rule_id for item in selected):
        return False
    if frame.risk_tags and any(_rule_contains_risk(rule, tag) for tag in frame.risk_tags):
        return True
    if rule.actions != selected[0].actions:
        return True
    if rule.stages != selected[0].stages and ("全阶段" in rule.stages or "任意" in rule.stages):
        return True
    if rule.contact_rounds != selected[0].contact_rounds and "任意" in rule.contact_rounds:
        return True
    return False


def _rule_contains_risk(rule: BusinessRule, risk_tag: str) -> bool:
    text = "\n".join([rule.decision, *rule.rationale, *rule.allowed_items, *rule.blocked_scripts])
    risk_terms = {
        "隐私披露": ("联系人", "披露", "敏感信息"),
        "敏感信息": ("资产", "房产", "车产", "存款", "股票", "基金"),
        "虚假法务": ("法院", "冻结", "诉讼", "律所", "法务"),
        "不实承诺": ("承诺", "减免", "停催", "撤案"),
        "施压表达": ("威胁", "施压", "马上", "必须"),
    }.get(risk_tag, (risk_tag,))
    return any(term in text for term in risk_terms)


def _render_rule_answer(state: RagWorkflowState, frame: BusinessFrame, rules: list[BusinessRule]) -> str:
    primary = rules[0]
    sources = build_source_text(state.sources)
    if not state.sources:
        rule_sources = _dedupe_text([source for rule in rules for source in rule.sources])
        sources = "引用来源：\n" + "\n".join(f"{index}. {source}" for index, source in enumerate(rule_sources, start=1))

    rationale = "\n".join(f"{index}. {item}" for index, item in enumerate(_merged_items(rule.rationale for rule in rules), start=1))
    allowed = "\n".join(f"{index}. {item}" for index, item in enumerate(_merged_items(rule.allowed_items for rule in rules), start=1))
    blocked = "\n".join(f"{index}. {item}" for index, item in enumerate(_merged_items(rule.blocked_scripts for rule in rules), start=1))
    action_steps = "\n".join(
        f"{index}. {item}" for index, item in enumerate(_build_action_steps(frame), start=1)
    )
    rule_scope = _rule_scope_text(frame, rules)

    scene_lines = [
        f"1. 阶段：{frame.stage or '未明确'}。",
        f"2. 对象：{frame.subject or '未明确'}。",
        f"3. 行为类型：{frame.action or '未明确'}。",
    ]
    if frame.contact_round:
        scene_lines.append(f"{len(scene_lines) + 1}. 通话轮次：{frame.contact_round}。")
    if frame.customer_emotion:
        scene_lines.append(f"{len(scene_lines) + 1}. 客户情绪：{frame.customer_emotion}。")
    if frame.information:
        scene_lines.append(f"{len(scene_lines) + 1}. 信息类型：{'、'.join(frame.information)}。")
    if frame.risk_tags:
        scene_lines.append(f"{len(scene_lines) + 1}. 风险标签：{'、'.join(frame.risk_tags)}。")

    return (
        f"结论\n{primary.decision}。\n\n"
        f"场景识别\n{chr(10).join(scene_lines)}\n\n"
        f"适用规则\n{rule_scope}\n\n"
        f"判断依据\n{rationale}\n\n"
        f"{_advice_title(frame.action)}\n{allowed}\n\n"
        f"操作拆解\n{action_steps}\n\n"
        f"不建议说法\n{blocked}\n\n"
        "需要确认事项\n"
        "1. 以当前系统记录中的逾期阶段、客户身份和授权状态为准。\n"
        "2. 涉及法务、联系人、资产、减免、停催等内容时，必要时提交主管或合规复核。\n\n"
        f"{sources}"
    )


def _merged_items(groups) -> list[str]:
    items: list[str] = []
    for group in groups:
        for item in group:
            if item and item not in items:
                items.append(item)
    return items


def _dedupe_text(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result


def _rule_scope_text(frame: BusinessFrame, rules: list[BusinessRule]) -> str:
    scopes: list[str] = []
    if frame.stage:
        scopes.append(f"{frame.stage} 阶段")
    if frame.action:
        scopes.append(frame.action)
    if frame.subject:
        scopes.append(f"{frame.subject}沟通")
    if frame.contact_round:
        scopes.append(frame.contact_round)
    if frame.customer_emotion:
        scopes.append(frame.customer_emotion)
    if frame.risk_tags:
        scopes.append("、".join(frame.risk_tags))
    if not scopes:
        scopes.append("通用业务边界")
    if len(rules) > 1:
        scopes.append("补充合规边界")
    return "、".join(scopes)


def _advice_title(action: str) -> str:
    if action in {"核资", "资产摸底"}:
        return "可询问方向"
    return "沟通重点"


def _build_action_steps(frame: BusinessFrame) -> list[str]:
    steps: list[str] = []

    if frame.subject == "联系人":
        steps.append("先确认联系人身份、关系和是否方便沟通，只做必要核实或转达。")
    elif frame.contact_round == "首通":
        steps.append("开场先说明身份和来意，再确认客户是否方便继续沟通。")
    elif frame.contact_round == "复通":
        steps.append("先回顾当前系统记录和上次沟通结果，再进入本次要确认的问题。")
    else:
        steps.append("先核对当前阶段、客户身份和系统记录，避免直接进入施压表达。")

    if frame.customer_emotion in {"轻度不满", "不满"}:
        steps.append("先回应客户的不满情绪，说明会记录反馈，再把话题拉回当前可处理事项。")
    elif frame.customer_emotion == "投诉":
        steps.append("先确认客户认为被骚扰或准备投诉的触发点，记录后按投诉风险流程复核。")
    elif frame.customer_emotion == "辱骂":
        steps.append("保持稳定语气，不回怼、不激化；无法继续沟通时按规则升级或结束。")

    if frame.action == "阶段沟通" and frame.stage == "D4-D6":
        steps.append("D4-D6 重点放在情况了解、还款意愿和可处理时间，提醒强度保持温和。")
    elif frame.action == "阶段沟通" and frame.stage == "D10-D15":
        steps.append("D10-D15 可以把目标推进到明确处理安排、具体日期和持续跟进动作。")
    elif frame.action == "核资":
        steps.append("核资问题围绕收入、工资、可调配资金、可处理金额和具体日期展开。")
    elif frame.action == "资产摸底":
        steps.append("资产相关信息只能服务于还款能力判断，避免把资产问题表达成威胁或法务后果。")
    elif frame.action == "联系人沟通":
        steps.append("联系人场景坚持最小披露，不提借款金额、逾期天数、催收压力或诉讼威胁。")
    elif frame.action == "法务表达":
        steps.append("法务表达只说真实流程和可核实状态，不预告确定冻结、移交或诉讼时间。")
    elif frame.action == "承诺":
        steps.append("涉及减免、停催、撤案、消除记录等事项，只能记录诉求并按权限核对。")
    else:
        steps.append("把沟通目标落到下一步动作，不把内部判断或历史信息当作对客户施压的依据。")

    steps.append("收口时只确认客户可配合的事项、明确日期和后续跟进方式，结果以系统记录和政策为准。")
    return steps


def _rule_types() -> set[str]:
    return {question_type for rule in load_business_rules() for question_type in rule.question_types}


def _rule_from_dict(item: dict[str, Any]) -> BusinessRule:
    return BusinessRule(
        rule_id=str(item["rule_id"]),
        question_types=tuple(item.get("question_types", [])),
        stages=tuple(item.get("stages", ["任意"])),
        actions=tuple(item.get("actions", ["任意"])),
        subjects=tuple(item.get("subjects", ["任意"])),
        contact_rounds=tuple(item.get("contact_rounds", ["任意"])),
        emotions=tuple(item.get("emotions", ["任意"])),
        decision=str(item.get("decision", "")),
        rationale=tuple(item.get("rationale", [])),
        allowed_items=tuple(item.get("allowed_items", [])),
        blocked_scripts=tuple(item.get("blocked_scripts", [])),
        sources=tuple(item.get("sources", [])),
        priority=int(item.get("priority", 50)),
    )


def _matches_dimension(value: str, candidates: tuple[str, ...]) -> bool:
    if not candidates:
        return True
    if "任意" in candidates or "全阶段" in candidates:
        return True
    if not value:
        return False
    return value in candidates


def _requires_specific_value(candidates: tuple[str, ...]) -> bool:
    return bool(candidates) and "任意" not in candidates and "全阶段" not in candidates
