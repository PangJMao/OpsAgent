from pathlib import Path
import sys
import types
import zipfile

from ops_agent.config import settings
from ops_agent.models import Chunk, RetrievalHit
from ops_agent.services import BgeReranker, LocalVectorStore, RagService, chunk_document, load_text_document
from ops_agent.services.document_service import normalize_to_markdown


def test_ingest_and_ask_by_markdown_heading(tmp_path: Path) -> None:
    document_path = tmp_path / "policy.md"
    document_path.write_text(
        "# 售后政策\n\n## 高级客户\n\n高级客户的售后响应时间为 4 小时内。",
        encoding="utf-8",
    )

    index_file = tmp_path / "index.db"
    pipeline = RagService(vector_store=LocalVectorStore(index_file=index_file))
    ingest_result = pipeline.ingest(document_path)

    assert ingest_result["chunk_count"] == 2
    assert ingest_result["strategy_counts"] == {"markdown_heading": 2}

    answer_pipeline = RagService(vector_store=LocalVectorStore(index_file=index_file))
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


def test_docx_normalizes_to_markdown(tmp_path: Path) -> None:
    document_path = tmp_path / "policy.docx"
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        "<w:p><w:r><w:t>高级客户售后响应时间为 4 小时内。</w:t></w:r></w:p>"
        "</w:body>"
        "</w:document>"
    )
    with zipfile.ZipFile(document_path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    document = load_text_document(document_path)

    assert document.metadata["source_format"] == "docx"
    assert "高级客户售后响应时间" in document.content


def test_xlsx_normalizes_to_markdown(tmp_path: Path) -> None:
    document_path = tmp_path / "policy.xlsx"
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheets><sheet name="售后政策" sheetId="1" r:id="rId1" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/></sheets>'
        "</workbook>"
    )
    shared_strings_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<si><t>客户等级</t></si><si><t>响应时间</t></si><si><t>高级客户</t></si><si><t>4 小时</t></si>"
        "</sst>"
    )
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
        '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2" t="s"><v>3</v></c></row>'
        "</sheetData>"
        "</worksheet>"
    )
    with zipfile.ZipFile(document_path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/sharedStrings.xml", shared_strings_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)

    normalized = normalize_to_markdown(document_path)

    assert normalized.metadata["source_format"] == "xlsx"
    assert "## 售后政策" in normalized.markdown
    assert "高级客户 | 4 小时" in normalized.markdown


def test_low_confidence_refuses(tmp_path: Path) -> None:
    index_file = tmp_path / "index.db"
    pipeline = RagService(vector_store=LocalVectorStore(index_file=index_file))
    answer = pipeline.ask("完全不存在的知识库问题")

    assert answer.refused is True
    assert answer.confidence < settings.min_relevance_score


class FakeVectorStore:
    def __init__(self) -> None:
        self.requested_top_k = 0

    def search(self, query: str, top_k: int = 4) -> list[RetrievalHit]:
        self.requested_top_k = top_k
        return [
            RetrievalHit(
                chunk=Chunk(
                    chunk_id=f"chunk-{index}",
                    document_id="doc",
                    title="测试制度",
                    text=f"测试知识片段 {index}",
                    start_char=0,
                    end_char=10,
                    metadata={"heading_path": ["测试制度"]},
                ),
                score=1.0 - (index * 0.01),
            )
            for index in range(12)
        ]

    def count(self) -> int:
        return 12


class FakeReranker:
    def __init__(self) -> None:
        self.input_count = 0
        self.requested_top_k = 0

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int = 3) -> list[RetrievalHit]:
        self.input_count = len(hits)
        self.requested_top_k = top_k
        return hits[:top_k]


def test_rag_retrieves_top_12_then_reranks_to_top_3() -> None:
    store = FakeVectorStore()
    reranker = FakeReranker()
    pipeline = RagService(vector_store=store, reranker=reranker)  # type: ignore[arg-type]

    answer = pipeline.ask("测试制度怎么执行？")

    assert store.requested_top_k == 12
    assert reranker.input_count == 12
    assert reranker.requested_top_k == 3
    assert answer.refused is False
    assert len(answer.citations) == 3


def test_bge_reranker_uses_flag_embedding_scores(monkeypatch) -> None:
    class FakeFlagReranker:
        def __init__(self, model_name: str, use_fp16: bool = True) -> None:
            self.model_name = model_name
            self.use_fp16 = use_fp16

        def compute_score(self, pairs: list[list[str]]) -> list[float]:
            assert pairs == [["问题", "片段 A"], ["问题", "片段 B"], ["问题", "片段 C"]]
            return [0.2, 0.9, 0.4]

    fake_module = types.SimpleNamespace(FlagReranker=FakeFlagReranker)
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_module)
    hits = [
        RetrievalHit(
            chunk=Chunk(
                chunk_id=f"chunk-{label}",
                document_id="doc",
                title="测试制度",
                text=f"片段 {label}",
                start_char=0,
                end_char=10,
            ),
            score=0.1,
        )
        for label in ["A", "B", "C"]
    ]

    ranked = BgeReranker(model_name="BAAI/bge-reranker-base").rerank("问题", hits, top_k=2)

    assert [hit.chunk.chunk_id for hit in ranked] == ["chunk-B", "chunk-C"]


def test_reingest_replaces_old_chunks_for_same_document(tmp_path: Path) -> None:
    document_path = tmp_path / "policy.md"
    document_path.write_text("# 售后政策\n\n## 高级客户\n\n高级客户需要 4 小时内响应。", encoding="utf-8")

    index_file = tmp_path / "index.db"
    pipeline = RagService(vector_store=LocalVectorStore(index_file=index_file))
    pipeline.ingest(document_path)

    store = LocalVectorStore(index_file=index_file)
    assert store.count() == 2

    pipeline = RagService(vector_store=store)
    pipeline.ingest(document_path)

    assert store.count() == 2
