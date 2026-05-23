from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

from ops_agent.models import RagAnswer
from ops_agent.services.rag_service import RagService


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_answer_contains: str = ""
    expect_refused: bool = False
    require_citation: bool = True


@dataclass(frozen=True)
class EvaluationResult:
    question: str
    passed: bool
    refused: bool
    confidence: float
    citation_count: int
    issues: list[str]


class AnswerService(Protocol):
    def ask(self, question: str) -> RagAnswer:
        ...


class EvaluationService:
    """Runs a small deterministic eval suite against the RAG answer path."""

    def __init__(self, answer_service: AnswerService | None = None) -> None:
        self.answer_service = answer_service or RagService()

    def run(self, cases: list[EvaluationCase]) -> dict[str, object]:
        results = [self._evaluate_case(case) for case in cases]
        total = len(results)
        passed = sum(1 for result in results if result.passed)
        citation_hits = sum(1 for result in results if result.citation_count > 0)
        refusal_matches = sum(
            1 for case, result in zip(cases, results, strict=True) if case.expect_refused == result.refused
        )
        return {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "citation_hit_rate": round(citation_hits / total, 4) if total else 0.0,
            "refusal_match_rate": round(refusal_matches / total, 4) if total else 0.0,
            "results": [asdict(result) for result in results],
        }

    def _evaluate_case(self, case: EvaluationCase) -> EvaluationResult:
        answer = self.answer_service.ask(case.question)
        issues: list[str] = []
        if answer.refused != case.expect_refused:
            issues.append("refusal_mismatch")
        if case.expected_answer_contains and case.expected_answer_contains not in answer.answer:
            issues.append("expected_text_missing")
        if case.require_citation and not case.expect_refused and not answer.citations:
            issues.append("citation_missing")
        return EvaluationResult(
            question=case.question,
            passed=not issues,
            refused=answer.refused,
            confidence=answer.confidence,
            citation_count=len(answer.citations),
            issues=issues,
        )
