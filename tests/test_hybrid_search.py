from ops_agent.models import Chunk, RetrievalHit
from ops_agent.config.retrieval_config import DEFAULT_RETRIEVAL_CONFIG
from ops_agent.services import RagService, answer_to_dict
from ops_agent.services.retriever.hybrid_retriever import HybridRetriever, hybrid_rank, merge_results, normalize_scores
from ops_agent.services.retriever.keyword_retriever import KeywordRetriever
from ops_agent.services.retriever.metadata_filter import MetadataFilter
from ops_agent.services.retriever.query_rewriter import QueryRewriter
from ops_agent.services.retriever.schema import HybridCandidate
from ops_agent.services.retriever.vector_retriever import VectorRetriever


def _hit(chunk_id: str, text: str, score: float = 0.8, **metadata) -> RetrievalHit:
    return RetrievalHit(
        Chunk(chunk_id, "doc", metadata.get("doc_name", "话术-沟通话术四大框架(2)"), text, 0, 10, metadata),
        score,
    )


class FakeStore:
    def __init__(self) -> None:
        self.rows = [
            _hit(
                "a",
                "客户骂人时需要先安抚，保持文明用语，正面回应客户问题。",
                0.82,
                doc_name="话术-沟通话术四大框架(2)",
                doc_type="沟通话术",
                sheet_name="Sheet3",
                topic="客户安抚",
                business_scene="客户不满",
                risk_level="medium",
            ),
            _hit(
                "b",
                "客户愿意沟通时，可以具体认可鼓励，但不能作出不实承诺。",
                0.78,
                doc_name="话术-沟通话术四大框架(2)",
                doc_type="沟通话术",
                sheet_name="Sheet3",
                topic="认可鼓励",
                business_scene="还款意愿",
                risk_level="medium",
            ),
            _hit(
                "c",
                "D4-D6 阶段以中性温和、信息核实为主，D7-D9 可适当加强提醒。",
                0.76,
                doc_name="核资话术分层建议",
                doc_type="沟通话术",
                sheet_name="阶段话术",
                topic="分阶段沟通",
                applicable_stage="D4-D6",
                risk_level="medium",
            ),
            _hit(
                "d",
                "联系人沟通要注意边界，不得泄露不必要信息；黑名单联系人应按规则处理。",
                0.74,
                doc_name="法务话术",
                doc_type="法务话术",
                sheet_name="联系人",
                topic="联系人沟通",
                business_scene="联系人",
                risk_level="high",
            ),
        ]

    def search(self, query: str, top_k: int = 10):
        return self.rows[:top_k]

    def keyword_search(self, query: str, top_k: int = 10):
        terms = [term for term in ["安抚", "文明用语", "客户骂人", "D4-D6", "联系人", "黑名单", "承诺"] if term in query]
        hits = []
        for row in self.rows:
            if any(term in row.chunk.text or term in str(row.chunk.metadata) for term in terms):
                hits.append(RetrievalHit(row.chunk, 2.0 + len(terms)))
        return hits[:top_k]


class FakeReranker:
    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int = 8):
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]


def test_query_rewrite_customer_communication() -> None:
    queries = QueryRewriter().rewrite("与客户沟通时有哪些技巧", "communication_script")
    joined = " ".join(queries)

    for word in ["安抚", "认可鼓励", "敏感词", "语速", "承诺", "合规"]:
        assert word in joined
    assert len(queries) <= 6


def test_hybrid_search_merge() -> None:
    chunk = _hit("same", "客户沟通需要安抚。")
    merged = merge_results(
        [
            HybridCandidate(hit=chunk, vector_score=0.7, sources=["vector"]),
            HybridCandidate(hit=chunk, keyword_score=3.0, matched_keywords=["安抚"], sources=["keyword"]),
        ]
    )
    normalize_scores(merged)
    ranked = hybrid_rank(merged, "客户沟通 安抚", DEFAULT_RETRIEVAL_CONFIG)

    assert len(ranked) == 1
    assert set(ranked[0].sources) == {"vector", "keyword"}
    assert ranked[0].hybrid_score > 0


def test_keyword_search_exact_terms() -> None:
    results = KeywordRetriever(FakeStore()).retrieve(["客户骂人怎么处理 安抚 文明用语"], top_k=10)

    assert results
    assert any("安抚" in result.matched_keywords for result in results)


def test_metadata_filter() -> None:
    candidate = HybridCandidate(hit=_hit("stage", "阶段话术", applicable_stage="D4-D6", doc_type="沟通话术"))
    score = MetadataFilter().score(candidate, "D4-D6 客户怎么沟通", "communication_script")

    assert score > 0.6


def test_deduplicate_same_sheet() -> None:
    result = HybridRetriever(VectorRetriever(FakeStore()), KeywordRetriever(FakeStore()), FakeReranker()).retrieve(
        "与客户沟通时有哪些技巧", "communication_script"
    )
    sheet_counts = {}
    for hit in result.hits:
        sheet = hit.chunk.metadata.get("sheet_name")
        sheet_counts[sheet] = sheet_counts.get(sheet, 0) + 1

    assert max(sheet_counts.values()) <= 3


def test_low_confidence_refuse() -> None:
    class LowStore(FakeStore):
        def search(self, query: str, top_k: int = 10):
            return [_hit("low", "无关内容", 0.1)]

        def keyword_search(self, query: str, top_k: int = 10):
            return []

    answer = RagService(vector_store=LowStore(), reranker=FakeReranker(), llm=type("Llm", (), {"enabled": False})()).ask("知识库完全无关的问题")

    assert answer.refused is True


def test_final_citation_hide_internal_ids() -> None:
    answer = RagService(vector_store=FakeStore(), reranker=FakeReranker(), llm=type("Llm", (), {"enabled": False})()).ask("与客户沟通时有哪些技巧")
    payload = answer_to_dict(answer)
    rendered = str(payload)

    assert "chunk_id" not in rendered
    assert "hybrid_score" not in answer.answer
    assert "vector_score" not in answer.answer
    assert "rerank_score" not in answer.answer


def test_customer_communication_answer_quality() -> None:
    answer = RagService(vector_store=FakeStore(), reranker=FakeReranker(), llm=type("Llm", (), {"enabled": False})()).ask("与客户沟通时有哪些技巧？")

    assert answer.refused is False
    for phrase in ["身份", "阶段", "语气", "安抚", "认可", "承诺", "联系人", "引用来源"]:
        assert phrase in answer.answer
