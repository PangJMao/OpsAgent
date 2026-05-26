from __future__ import annotations

import re

from ops_agent.services.retriever.schema import HybridCandidate


class MetadataFilter:
    """Soft metadata scoring; keeps recall broad while favoring intent-matched evidence."""

    def score(self, candidate: HybridCandidate, question: str, question_type: str) -> float:
        metadata = candidate.hit.chunk.metadata
        score = 0.3
        doc_type = str(metadata.get("doc_type") or "")
        topic = str(metadata.get("topic") or "")
        scene = str(metadata.get("business_scene") or "")
        stage = str(metadata.get("applicable_stage") or "")
        risk = str(metadata.get("risk_level") or "low")
        section = str(metadata.get("section") or "")
        text = f"{doc_type} {topic} {scene} {stage} {section} {candidate.hit.chunk.text}"

        if question_type in {"communication_script", "new_collector", "customer_abuse", "complaint_threat", "stage_script"}:
            if _contains_any(text, "沟通", "话术", "安抚", "认可", "客户", "催收", "娌熐?", "璇濇湳"):
                score += 0.30
        if question_type in {"legal_compliance", "asset_inquiry", "false_commitment", "contact_boundary", "privacy_boundary"}:
            if _contains_any(text, "法务", "合规", "禁止", "不得", "风险", "联系人", "隐私", "诉讼", "冻结"):
                score += 0.35
        if question_type in {"clarification_medical", "clarification_overseas", "reduction_request", "negotiation_stage"}:
            if _contains_any(text, "人工核对", "系统记录", "政策", "当前", "授权", "边界", "不得"):
                score += 0.20

        stage_in_question = _stage_from_question(question)
        if stage_in_question and stage_in_question in text:
            score += 0.35
        if question_type == "stage_script" and stage:
            score += 0.15

        for marker in (
            "D4-D6",
            "D7-D9",
            "D10-D15",
            "联系人",
            "紧急联系人",
            "预留联系人",
            "黑名单",
            "敏感词",
            "投诉",
            "承诺",
            "合规",
            "房产",
            "车产",
            "存款",
            "股票",
            "基金",
            "鑱旂郴浜?",
            "榛戝悕鍗?",
            "鏁忔劅璇?",
            "鎶曡瘔",
            "鎵胯",
            "鍚堣",
        ):
            if marker in question and marker in text:
                score += 0.20

        if not _contains_any(question, "法务", "诉讼", "冻结", "法院", "投诉", "骚扰", "联系人", "娉曞姟") and risk == "high":
            score -= 0.10
        return min(max(score, 0.0), 1.0)

    def apply(self, candidates: list[HybridCandidate], question: str, question_type: str) -> list[HybridCandidate]:
        for candidate in candidates:
            candidate.metadata_score = self.score(candidate, question, question_type)
        return candidates


def _contains_any(text: str, *terms: str) -> bool:
    return any(term in text for term in terms)


def _stage_from_question(question: str) -> str:
    match = re.search(r"D\d+\s*[-~至到鑷冲埌]\s*D?\d+", question, re.I)
    return match.group(0) if match else ""
