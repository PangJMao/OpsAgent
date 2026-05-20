from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ops_agent.core.config import settings
from ops_agent.ingestion.chunker import chunk_document
from ops_agent.ingestion.loaders import (
    load_text_document,
    persist_normalized_markdown,
    persist_source_document,
)
from ops_agent.observability import TraceRecorder
from ops_agent.retrieval import LocalVectorStore
from ops_agent.schemas import Chunk, Citation, RagAnswer, RetrievalHit


class RagPipeline:
    """第一阶段 RAG 应用服务。"""

    def __init__(
        self,
        vector_store: LocalVectorStore | None = None,
        recorder: TraceRecorder | None = None,
    ) -> None:
        self.vector_store = vector_store or LocalVectorStore()
        self.recorder = recorder or TraceRecorder()

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
            span.update(
                {
                    "stored_path": str(stored_path),
                    "normalized_path": str(normalized_path),
                }
            )

        with self.recorder.span("document.chunk", {"document_id": document.document_id}) as span:
            chunks = chunk_document(document)
            span.update(
                {
                    "chunk_count": len(chunks),
                    "strategy_counts": _strategy_counts(chunks),
                }
            )

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

        with self.recorder.span("retriever.search", {"top_k": settings.top_k}) as span:
            hits = self.vector_store.search(normalized_question, top_k=settings.top_k)
            span.update(
                {
                    "hit_count": len(hits),
                    "best_score": round(hits[0].score, 4) if hits else 0.0,
                    "chunk_ids": [hit.chunk.chunk_id for hit in hits],
                    "heading_paths": [hit.chunk.metadata.get("heading_path", []) for hit in hits],
                }
            )

        # 置信度门控是第一道反幻觉控制。
        # 当证据不足时系统直接拒答，避免用模型先验补全企业内部事实。
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

    def _compose_answer(
        self,
        question: str,
        hits: list[RetrievalHit],
        citations: list[Citation],
    ) -> str:
        evidence_lines = []
        for index, hit in enumerate(hits, start=1):
            if hit.score <= 0:
                continue
            heading_path = hit.chunk.metadata.get("heading_path") or []
            heading_label = f"（章节：{' > '.join(heading_path)}）" if heading_path else ""
            evidence_lines.append(f"[{index}]{heading_label}\n{hit.chunk.text}")

        evidence = "\n\n".join(evidence_lines)
        sources = _format_user_visible_sources(citations)
        return (
            f"根据当前知识库中与“{question}”最相关的资料，可以归纳如下：\n\n"
            f"{evidence}\n\n"
            "以上内容仅基于已检索到的知识库片段生成。\n\n"
            f"{sources}"
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


def _format_user_visible_sources(citations: list[Citation]) -> str:
    if not citations:
        return "引用来源：无。"

    lines = ["引用来源："]
    for index, citation in enumerate(citations, start=1):
        heading = " > ".join(citation.heading_path) if citation.heading_path else "未标注章节"
        lines.append(
            f"{index}. 文档：{citation.title}；章节：{heading}；"
            f"片段：{citation.chunk_id}；相关度：{citation.score:.4f}"
        )
    return "\n".join(lines)


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
