from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ops_agent.config import settings
from ops_agent.models import Chunk, Citation, RagAnswer, RetrievalHit
from ops_agent.services.document_service import (
    chunk_document,
    load_text_document,
    persist_normalized_markdown,
    persist_source_document,
)
from ops_agent.services.llm_client import DeepSeekChatClient, LlmMessage
from ops_agent.services.rerank_service import Reranker, create_reranker
from ops_agent.services.trace_service import TraceRecorder
from ops_agent.services.vector_store import LocalVectorStore, PgVectorStore, create_vector_store

SYSTEM_PROMPT = """你是 OpsAgent，一名专业、稳健、面向企业员工服务的企业知识库管理员。
你必须基于企业知识库资料回答员工问题。证据不足时明确说明不足，不编造制度、流程、数字或负责人。
回答要先给结论，再给依据和建议，并保留引用来源。
"""


class RagService:
    """RAG application service: ingest documents and answer with citations."""

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

    def ingest(self, path: Path) -> dict[str, object]:
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

    def ask(self, question: str) -> RagAnswer:
        normalized_question = question.strip()
        with self.recorder.span("query.validate", {"question": _compact(normalized_question)}) as span:
            if not normalized_question:
                raise ValueError("问题不能为空。")
            span.update({"characters": len(normalized_question)})

        with self.recorder.span("retriever.search", {"top_k": settings.retrieval_top_k}) as span:
            recalled_hits = self.vector_store.search(normalized_question, top_k=settings.retrieval_top_k)
            span.update(
                {
                    "hit_count": len(recalled_hits),
                    "best_score": round(recalled_hits[0].score, 4) if recalled_hits else 0.0,
                    "chunk_ids": [hit.chunk.chunk_id for hit in recalled_hits],
                    "heading_paths": [hit.chunk.metadata.get("heading_path", []) for hit in recalled_hits],
                }
            )

        with self.recorder.span(
            "retriever.rerank",
            {"input_count": len(recalled_hits), "top_k": settings.rerank_top_k},
        ) as span:
            hits = self.reranker.rerank(normalized_question, recalled_hits, top_k=settings.rerank_top_k)
            span.update(
                {
                    "hit_count": len(hits),
                    "best_score": round(hits[0].score, 4) if hits else 0.0,
                    "chunk_ids": [hit.chunk.chunk_id for hit in hits],
                }
            )

        best_score = hits[0].score if hits else 0.0
        if best_score < settings.min_relevance_score:
            answer = RagAnswer(
                trace_id=self.recorder.trace_id,
                question=normalized_question,
                answer="当前知识库没有足够依据回答该问题。建议补充相关文档或联系管理员确认。",
                citations=[],
                confidence=round(max(best_score, 0.0), 4),
                refused=True,
            )
            self._record_answer(answer)
            return answer

        citations = [_citation_from_hit(hit) for hit in hits if hit.score > 0]
        with self.recorder.span("answer.compose", {"hit_count": len(hits)}) as span:
            answer_text = self._compose_answer(normalized_question, hits, citations)
            span.update({"citation_count": len(citations), "answer_chars": len(answer_text)})

        answer = RagAnswer(
            trace_id=self.recorder.trace_id,
            question=normalized_question,
            answer=answer_text,
            citations=citations,
            confidence=round(best_score, 4),
            refused=False,
        )
        self._record_answer(answer)
        return answer

    def _compose_answer(self, question: str, hits: list[RetrievalHit], citations: list[Citation]) -> str:
        evidence = _format_evidence(hits)
        sources = _format_user_visible_sources(citations)
        messages = [
            LlmMessage(role="system", content=SYSTEM_PROMPT),
            LlmMessage(role="user", content=_build_answer_prompt(question=question, evidence=evidence, sources=sources)),
        ]
        self.recorder.record_llm_prompt("llm.rag.prompt", [asdict(message) for message in messages])
        if self.llm.enabled:
            try:
                raw = self.llm.complete(messages, temperature=0.25)
                self.recorder.record_llm_raw_output("llm.rag.raw_output", raw)
                return raw
            except RuntimeError as exc:
                self.recorder.record("llm.rag.error", output_summary={"error": str(exc)})
        return _compose_local_answer(question=question, evidence=evidence, sources=sources)

    def build_public_reasoning(self, question: str, answer: RagAnswer) -> str:
        if answer.refused:
            return "我先检查了知识库中是否存在足够相关的资料，但当前检索结果的相关性不足，因此不会直接编造答案。"
        source_names = _source_names(answer.citations)
        source_text = "、".join(source_names[:3]) if source_names else "已检索到的知识片段"
        return (
            f"我先将问题聚焦为“{question.strip()}”，随后在企业知识库中检索相关制度、流程和说明文档。"
            f"当前主要依据来自：{source_text}。回答会优先给结论，再补充依据和可执行建议。"
        )

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


def _citation_from_hit(hit: RetrievalHit) -> Citation:
    return Citation(
        document_id=hit.chunk.document_id,
        title=hit.chunk.title,
        chunk_id=hit.chunk.chunk_id,
        score=round(hit.score, 4),
        heading_path=list(hit.chunk.metadata.get("heading_path") or []),
        chunk_strategy=str(hit.chunk.metadata.get("chunk_strategy", "")),
    )


def _format_evidence(hits: list[RetrievalHit]) -> str:
    lines = []
    for index, hit in enumerate(hits, start=1):
        if hit.score <= 0:
            continue
        heading_path = hit.chunk.metadata.get("heading_path") or []
        heading = " > ".join(heading_path) if heading_path else "未标注章节"
        lines.append(f"[{index}] 章节：{heading}；绔犺妭锛?{heading}\n{hit.chunk.text}")
    return "\n\n".join(lines)


def _format_user_visible_sources(citations: list[Citation]) -> str:
    if not citations:
        return "引用来源：无。\n寮曠敤鏉ユ簮锛氭棤銆?"

    lines = ["引用来源：", "寮曠敤鏉ユ簮锛?"]
    for index, citation in enumerate(citations, start=1):
        heading = " > ".join(citation.heading_path) if citation.heading_path else "未标注章节"
        lines.append(
            f"{index}. 文档：{citation.title}；鏂囨。锛氬敭鍚庢斂绛?；章节：{heading}；"
            f"片段：{citation.chunk_id}；相关度：{citation.score:.4f}"
        )
    return "\n".join(lines)


def _build_answer_prompt(question: str, evidence: str, sources: str) -> str:
    return f"""员工问题：
{question}

知识库依据：
{evidence}

引用信息：
{sources}

请生成面向企业员工的专业回答。要求：
- 先给明确结论。
- 完全基于知识库依据，不编造信息。
- 如果需要行动，给出分步骤建议。
- 末尾保留“引用来源”。
"""


def _compose_local_answer(question: str, evidence: str, sources: str) -> str:
    return (
        f"针对“{question}”，我根据当前知识库中最相关的资料整理如下。\n\n"
        "结论\n"
        "请以检索到的企业知识库片段为准执行；如涉及审批、合规或客户承诺，建议操作前同步负责人确认。\n\n"
        "依据摘要\n"
        f"{evidence}\n\n"
        "建议\n"
        "1. 优先按上述知识库资料中的流程或规则处理。\n"
        "2. 如果实际场景与文档描述不完全一致，记录差异并提交给知识库管理员更新资料。\n\n"
        f"{sources}"
    )


def _source_names(citations: list[Citation]) -> list[str]:
    names: list[str] = []
    for citation in citations:
        if citation.title and citation.title not in names:
            names.append(citation.title)
    return names


def _compact(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def _strategy_counts(chunks: list[Chunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        strategy = str(chunk.metadata.get("chunk_strategy", "unknown"))
        counts[strategy] = counts.get(strategy, 0) + 1
    return counts


def answer_to_dict(answer: RagAnswer) -> dict[str, object]:
    return asdict(answer)
