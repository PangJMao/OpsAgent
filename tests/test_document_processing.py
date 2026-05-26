from pathlib import Path
import sys
import types
import zipfile

from ops_agent.config import settings
from ops_agent.models import Chunk, RetrievalHit
from ops_agent.services.document_processing.loaders import DocumentLoader
from ops_agent.services.document_processing.parsers import PdfParser
from ops_agent.services.document_processing.pipeline import IngestionPipeline
from ops_agent.services.document_processing.retrieval import HybridRetriever, HybridSearchConfig
from ops_agent.services.document_processing.vector_store import LocalDocumentVectorStore, PgDocumentVectorStore, create_document_vector_store
from ops_agent.services.vector_store import LocalVectorStore


def test_document_processing_pipeline_ingests_mvp_formats(tmp_path: Path) -> None:
    workdir = tmp_path / "workfile"
    workdir.mkdir()
    (workdir / "policy.md").write_text("# 售后政策\n\n## 高级客户\n\n4 小时内响应。", encoding="utf-8")
    (workdir / "note.txt").write_text("普通文本资料。", encoding="utf-8")
    (workdir / "inventory.csv").write_text("产品,价格,库存\niPhone,6999,12\n", encoding="utf-8")

    index_file = tmp_path / "index.db"
    pipeline = IngestionPipeline(vector_store=LocalDocumentVectorStore(LocalVectorStore(index_file=index_file)))
    result = pipeline.ingest_path(workdir)

    assert result.source_count == 3
    assert result.failed == {}
    assert result.chunk_count >= 3
    assert pipeline.vector_store.count() == result.chunk_count

    csv_chunks = [chunk for chunk in result.chunks if chunk.metadata["file_type"] == "csv"]
    assert csv_chunks
    assert csv_chunks[0].metadata["source"]
    assert csv_chunks[0].metadata["file_name"] == "inventory.csv"
    assert "content_hash" in csv_chunks[0].metadata
    assert "产品为iPhone" in "\n".join(chunk.text for chunk in csv_chunks)


def test_phase2_parses_pptx_html_and_image_fallback(tmp_path: Path) -> None:
    pptx_path = tmp_path / "deck.pptx"
    with zipfile.ZipFile(pptx_path, "w") as archive:
        archive.writestr(
            "ppt/slides/slide1.xml",
            '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
            'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            "<p:cSld><p:spTree><p:sp><p:txBody>"
            "<a:p><a:r><a:t>季度计划</a:t></a:r></a:p>"
            "<a:p><a:r><a:t>完成知识库升级</a:t></a:r></a:p>"
            "</p:txBody></p:sp></p:spTree></p:cSld></p:sld>",
        )
    html_path = tmp_path / "page.html"
    html_path.write_text(
        "<html><head><title>帮助中心</title></head><body><nav>忽略</nav><h1>入库指南</h1><p>支持网页正文。</p><footer>忽略</footer></body></html>",
        encoding="utf-8",
    )
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(b"not-a-real-image")

    loader = DocumentLoader()
    pptx = loader.load(pptx_path)
    html = loader.load(html_path)
    image = loader.load(image_path)

    assert pptx.file_type == "pptx"
    assert pptx.blocks[0].slide_number == 1
    assert pptx.blocks[0].metadata["slide_title"] == "季度计划"
    assert html.file_type == "html"
    assert html.blocks[0].heading_path == ["入库指南"]
    assert "支持网页正文" in html.blocks[0].text
    assert image.file_type == "image"
    assert image.blocks[0].metadata["image_name"] == "scan.png"


def test_legacy_doc_fallback_extracts_text(tmp_path: Path) -> None:
    doc_path = tmp_path / "legacy.doc"
    doc_path.write_bytes("旧版 Word 文档内容".encode("utf-16le"))

    document = DocumentLoader().load(doc_path)

    assert document.file_type == "doc"
    assert "旧版 Word 文档内容" in document.blocks[0].text


def test_docx_with_legacy_doc_content_uses_fallback(tmp_path: Path) -> None:
    docx_path = tmp_path / "renamed.docx"
    docx_path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + "旧版 Word 文档内容".encode("utf-16le"))

    document = DocumentLoader().load(docx_path)

    assert document.file_type == "doc"
    assert document.metadata["legacy_format"] is True


def test_document_vector_store_defaults_to_pgvector(monkeypatch) -> None:
    object.__setattr__(settings, "document_vector_provider", "pgvector")
    monkeypatch.setattr(
        "ops_agent.services.document_processing.vector_store.PgVectorStore",
        lambda: object(),
    )

    store = create_document_vector_store()

    assert isinstance(store, PgDocumentVectorStore)


def test_pdf_parser_uses_docling_when_available(tmp_path: Path, monkeypatch) -> None:
    class FakeDocument:
        def export_to_markdown(self) -> str:
            return "# 复杂 PDF\n\n| 指标 | 值 |\n| --- | --- |\n| SLA | 4 小时 |"

    class FakeConverter:
        def convert(self, path: str) -> object:
            return types.SimpleNamespace(document=FakeDocument())

    fake_module = types.SimpleNamespace(DocumentConverter=FakeConverter)
    monkeypatch.setitem(sys.modules, "docling", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "docling.document_converter", fake_module)
    object.__setattr__(settings, "pdf_parser_backend", "docling")

    document = PdfParser().parse(tmp_path / "complex.pdf")

    assert document.parser == "docling"
    assert document.metadata["layout_backend"] == "docling"
    assert "复杂 PDF" in document.blocks[0].text


def test_local_keyword_search_finds_exact_terms(tmp_path: Path) -> None:
    store = LocalVectorStore(index_file=tmp_path / "index.db")
    store.upsert_chunks(
        [
            Chunk("a", "doc", "A", "错误码 E1001 表示库存不足", 0, 10, {}),
            Chunk("b", "doc2", "B", "普通售后政策", 0, 10, {}),
        ]
    )

    hits = store.keyword_search("E1001", top_k=2)

    assert hits
    assert hits[0].chunk.chunk_id == "a"


def test_hybrid_retriever_merges_vector_and_keyword_scores() -> None:
    class FakeStore:
        def search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
            return [
                RetrievalHit(Chunk("semantic", "doc", "语义", "售后响应时间", 0, 10, {}), 0.9),
                RetrievalHit(Chunk("exact", "doc", "精确", "错误码 E1001 库存不足", 0, 10, {}), 0.1),
            ]

        def keyword_search(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
            return [RetrievalHit(Chunk("exact", "doc", "精确", "错误码 E1001 库存不足", 0, 10, {}), 3.0)]

    retriever = HybridRetriever(
        store=FakeStore(),
        reranker=types.SimpleNamespace(rerank=lambda query, hits, top_k: hits[:top_k]),
        config=HybridSearchConfig(vector_weight=0.4, bm25_weight=0.6, recall_multiplier=1),
    )

    hits = retriever.search("E1001", top_k=2, rerank=False)

    assert hits[0].chunk.chunk_id == "exact"
    assert hits[0].chunk.metadata["retrieval_mode"] == "hybrid"


def test_xlsx_parser_creates_structured_row_metadata(tmp_path: Path) -> None:
    from openpyxl import Workbook

    path = tmp_path / "talk.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet3"
    sheet.append(["问题", "标准答案", "适用场景", "合规要求", "适用阶段"])
    sheet.append(["客户不满怎么办", "先安抚再解释", "客户投诉", "不得威胁客户", "D4-D6"])
    workbook.save(path)

    document = DocumentLoader().load(path)
    row_blocks = [block for block in document.blocks if block.block_type == "table_row"]

    assert row_blocks
    assert "问题: 客户不满怎么办" in row_blocks[0].text
    assert row_blocks[0].metadata["business_scene"] == "客户投诉"
    assert row_blocks[0].metadata["risk_level"] == "high"
    assert row_blocks[0].metadata["applicable_stage"] == "D4-D6"
