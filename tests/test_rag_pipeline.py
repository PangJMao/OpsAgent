from pathlib import Path

from ops_agent.core.config import settings
from ops_agent.ingestion.chunker import chunk_document
from ops_agent.ingestion.loaders import load_text_document
from ops_agent.rag import RagPipeline
from ops_agent.retrieval import LocalVectorStore


def test_ingest_and_ask_by_markdown_heading(tmp_path: Path) -> None:
    document_path = tmp_path / "policy.md"
    document_path.write_text(
        "# 售后政策\n\n## 高级客户\n\n高级客户的售后响应时间为 4 小时内。",
        encoding="utf-8",
    )

    index_file = tmp_path / "index.db"
    pipeline = RagPipeline(vector_store=LocalVectorStore(index_file=index_file))
    ingest_result = pipeline.ingest(document_path)

    assert ingest_result["chunk_count"] == 2
    assert ingest_result["strategy_counts"] == {"markdown_heading": 2}

    answer_pipeline = RagPipeline(vector_store=LocalVectorStore(index_file=index_file))
    answer = answer_pipeline.ask("高级客户售后多久响应？")

    assert answer.refused is False
    assert "4 小时" in answer.answer
    assert "章节：" in answer.answer
    assert "引用来源：" in answer.answer
    assert "文档：售后政策" in answer.answer
    assert answer.citations
    assert answer.citations[0].heading_path


def test_long_markdown_section_uses_window_fallback(tmp_path: Path) -> None:
    document_path = tmp_path / "long.md"
    long_text = "高级客户需要 4 小时内响应。" * 120
    document_path.write_text(f"# 售后政策\n\n## 高级客户\n\n{long_text}", encoding="utf-8")

    document = load_text_document(document_path)
    chunks = chunk_document(document)

    assert len(chunks) > 1
    assert chunks[0].metadata["chunk_strategy"] == "markdown_heading"
    assert chunks[0].metadata["fallback_used"] is False
    assert chunks[0].metadata["heading_path"] == ["售后政策"]
    assert all(chunk.metadata["chunk_strategy"] == "markdown_heading_window_fallback" for chunk in chunks[1:])
    assert all(chunk.metadata["fallback_used"] is True for chunk in chunks[1:])
    assert all(chunk.metadata["heading_path"] == ["售后政策", "高级客户"] for chunk in chunks[1:])


def test_plain_text_normalizes_to_markdown(tmp_path: Path) -> None:
    document_path = tmp_path / "policy.txt"
    document_path.write_text("高级客户的售后响应时间为 4 小时内。", encoding="utf-8")

    document = load_text_document(document_path)
    chunks = chunk_document(document)

    assert document.content.startswith("# policy")
    assert chunks[0].metadata["chunk_strategy"] == "markdown_heading"
    assert chunks[0].metadata["heading_path"] == ["policy"]


def test_low_confidence_refuses(tmp_path: Path) -> None:
    index_file = tmp_path / "index.db"
    pipeline = RagPipeline(vector_store=LocalVectorStore(index_file=index_file))
    answer = pipeline.ask("完全不存在的知识库问题")

    assert answer.refused is True
    assert answer.confidence < settings.min_relevance_score


def test_reingest_replaces_old_chunks_for_same_document(tmp_path: Path) -> None:
    document_path = tmp_path / "policy.md"
    document_path.write_text("# 售后政策\n\n## 高级客户\n\n高级客户需要 4 小时内响应。", encoding="utf-8")

    index_file = tmp_path / "index.db"
    pipeline = RagPipeline(vector_store=LocalVectorStore(index_file=index_file))
    pipeline.ingest(document_path)

    store = LocalVectorStore(index_file=index_file)
    assert store.count() == 2

    pipeline = RagPipeline(vector_store=store)
    pipeline.ingest(document_path)

    assert store.count() == 2
