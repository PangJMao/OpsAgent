from pathlib import Path
import sys
import types
import zipfile

from ops_agent.config import settings
from ops_agent.models import Chunk, RetrievalHit
from ops_agent.services import BgeReranker, LocalKeywordReranker, LocalVectorStore, RagService, answer_to_dict, chunk_document, load_text_document
from ops_agent.services.rerank_service import create_reranker
from ops_agent.services.document_service import normalize_to_markdown
from ops_agent.services.rag_workflow import EnterpriseRagWorkflow
from ops_agent.services.document_processing.cleaners import clean_text, is_meaningful_text


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
    assert "《售后政策》" in answer.answer
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


class FakeHybridStore(FakeVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.keyword_requested_top_k = 0

    def search(self, query: str, top_k: int = 4) -> list[RetrievalHit]:
        self.requested_top_k = top_k
        return [
            RetrievalHit(
                chunk=Chunk(
                    chunk_id="semantic",
                    document_id="doc",
                    title="语义制度",
                    text="售后响应时间为 4 小时",
                    start_char=0,
                    end_char=10,
                    metadata={"heading_path": ["语义制度"]},
                ),
                score=0.9,
            )
        ]

    def keyword_search(self, query: str, top_k: int = 4) -> list[RetrievalHit]:
        self.keyword_requested_top_k = top_k
        return [
            RetrievalHit(
                chunk=Chunk(
                    chunk_id="exact",
                    document_id="doc",
                    title="错误码制度",
                    text="错误码 E1001 表示库存不足",
                    start_char=0,
                    end_char=10,
                    metadata={"heading_path": ["错误码制度"]},
                ),
                score=2.0,
            )
        ]


def test_rag_retrieves_top_12_then_reranks_to_top_3() -> None:
    store = FakeVectorStore()
    reranker = FakeReranker()
    pipeline = RagService(vector_store=store, reranker=reranker)  # type: ignore[arg-type]

    answer = pipeline.ask("测试制度怎么执行？")

    assert store.requested_top_k == 10
    assert reranker.input_count == 12
    assert reranker.requested_top_k == 8
    assert answer.refused is False
    assert len(answer.citations) == 3


def test_rag_uses_hybrid_retriever_when_keyword_search_is_available() -> None:
    store = FakeHybridStore()
    reranker = FakeReranker()
    pipeline = RagService(vector_store=store, reranker=reranker)  # type: ignore[arg-type]

    answer = pipeline.ask("E1001 是什么意思")

    assert store.requested_top_k == 10
    assert store.keyword_requested_top_k == 10
    assert reranker.input_count == 2
    assert answer.refused is False
    assert {citation.chunk_id for citation in answer.citations} == {"semantic", "exact"}


def test_local_answer_is_structured_for_business_question() -> None:
    class Store:
        def search(self, query: str, top_k: int = 4) -> list[RetrievalHit]:
            return [
                RetrievalHit(
                    Chunk(
                        "c1",
                        "doc",
                        "沟通话术",
                        "客户沟通时先确认客户诉求，复述问题，避免直接给结论。遇到异议时先回应顾虑，再给出依据和下一步方案。",
                        0,
                        10,
                        {"heading_path": ["客户沟通"]},
                    ),
                    0.9,
                )
            ]

        def count(self) -> int:
            return 1

    pipeline = RagService(
        vector_store=Store(),  # type: ignore[arg-type]
        reranker=FakeReranker(),
        llm=types.SimpleNamespace(enabled=False),
    )

    answer = pipeline.ask("与客户沟通时有哪些技巧")

    assert answer.refused is False
    assert "结论" in answer.answer
    assert "关键要点" in answer.answer
    assert "操作建议" in answer.answer
    assert "引用来源：" in answer.answer
    assert "先确认客户诉求" in answer.answer
    payload = answer_to_dict(answer)
    assert "chunk_id" not in str(payload["citations"])
    assert "score" not in str(payload["citations"])


def test_query_rewrite_expands_customer_communication_question() -> None:
    workflow = EnterpriseRagWorkflow(store=FakeHybridStore(), reranker=FakeReranker())  # type: ignore[arg-type]
    state = workflow.run("与客户沟通时有哪些技巧")

    assert state.question_type == "communication_script"
    assert any("客户不满" in query for query in state.queries)
    assert any("联系人沟通" in query for query in state.queries)


def test_low_relevance_refuses_answer() -> None:
    class LowStore:
        def search(self, query: str, top_k: int = 4) -> list[RetrievalHit]:
            return [RetrievalHit(Chunk("low", "doc", "低相关", "完全无关内容", 0, 10, {}), 0.2)]

        def keyword_search(self, query: str, top_k: int = 4) -> list[RetrievalHit]:
            return []

        def count(self) -> int:
            return 1

    answer = RagService(vector_store=LowStore(), reranker=FakeReranker(), llm=types.SimpleNamespace(enabled=False)).ask("与客户沟通时有哪些技巧")  # type: ignore[arg-type]

    assert answer.refused is True
    assert "暂不建议直接回答" in answer.answer


def test_clean_text_removes_mojibake_and_control_noise() -> None:
    text = clean_text("标题\n标题\n\x00  客户沟通   技巧  \n\n\n")

    assert text == "标题\n客户沟通 技巧"
    assert is_meaningful_text(text)


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


def test_local_reranker_provider_does_not_import_bge(monkeypatch) -> None:
    object.__setattr__(settings, "rerank_provider", "local")

    reranker = create_reranker()

    assert isinstance(reranker, LocalKeywordReranker)


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


def test_reingest_marks_old_chunks_deleted_and_logs(tmp_path: Path) -> None:
    document_path = tmp_path / "policy.md"
    document_path.write_text("# Policy\n\n## A\n\nfirst version", encoding="utf-8")

    index_file = tmp_path / "index.db"
    pipeline = RagService(vector_store=LocalVectorStore(index_file=index_file))
    pipeline.ingest(document_path)
    document_path.write_text("# Policy\n\n## A\n\nsecond version", encoding="utf-8")
    pipeline.ingest(document_path)

    store = LocalVectorStore(index_file=index_file)
    with store._connect() as connection:
        deleted_count = connection.execute("SELECT COUNT(*) AS total FROM chunks WHERE deleted = 1").fetchone()["total"]
        active_rows = connection.execute("SELECT text FROM chunks WHERE deleted = 0").fetchall()
        log_rows = connection.execute(
            "SELECT old_chunk_count, new_chunk_count, status FROM ingestion_logs ORDER BY log_id"
        ).fetchall()

    assert deleted_count == 2
    assert store.count() == 2
    assert all("second version" in row["text"] or "Policy" in row["text"] for row in active_rows)
    assert len(log_rows) == 2
    assert log_rows[-1]["old_chunk_count"] == 2
    assert log_rows[-1]["new_chunk_count"] == 2
    assert log_rows[-1]["status"] == "success"


def test_clear_all_marks_active_chunks_deleted(tmp_path: Path) -> None:
    store = LocalVectorStore(index_file=tmp_path / "index.db")
    store.upsert_chunks(
        [
            Chunk("a", "doc-a", "A", "沟通技巧", 0, 10, {"source": "a.md"}),
            Chunk("b", "doc-b", "B", "法务话术", 0, 10, {"source": "b.md"}),
        ]
    )

    deleted_count = store.clear_all()

    with store._connect() as connection:
        log = connection.execute("SELECT status, old_chunk_count FROM ingestion_logs ORDER BY log_id DESC").fetchone()
    assert deleted_count == 2
    assert store.count() == 0
    assert log["status"] == "cleared"
    assert log["old_chunk_count"] == 2


def test_rag_ingests_xlsx_with_structured_metadata(tmp_path: Path) -> None:
    from openpyxl import Workbook

    path = tmp_path / "communication.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "客户沟通"
    sheet.append(["问题", "标准答案", "适用场景", "合规要求"])
    sheet.append(["客户不满怎么办", "先安抚客户，再解释规则", "客户投诉", "不得威胁客户"])
    workbook.save(path)

    store = LocalVectorStore(index_file=tmp_path / "index.db")
    result = RagService(vector_store=store).ingest(path)
    hits = store.search("客户不满如何沟通", top_k=3)

    assert result["chunk_count"] >= 1
    assert hits
    assert hits[0].chunk.metadata["sheet_name"] == "客户沟通"
    assert hits[0].chunk.metadata["business_scene"] == "客户投诉"
    assert hits[0].chunk.metadata["risk_level"] == "high"
