from pathlib import Path

from ops_agent.services import AgentService, agent_answer_to_dict, chunk_document, load_text_document
from ops_agent.services.llm_client import LlmMessage
from ops_agent.services.vector_store import LocalVectorStore


def _index_policy(tmp_path: Path) -> LocalVectorStore:
    document_path = tmp_path / "policy.md"
    document_path.write_text(
        "# 售后政策\n\n## 高级客户\n\n高级客户的售后响应时间为 4 小时内。",
        encoding="utf-8",
    )
    document = load_text_document(document_path)
    chunks = chunk_document(document)
    store = LocalVectorStore(index_file=tmp_path / "agent_index.db")
    store.upsert_chunks(chunks)
    return store


def test_agent_runs_knowledge_qa_with_review_and_citations(tmp_path: Path) -> None:
    store = _index_policy(tmp_path)
    workflow = AgentService(vector_store=store)

    answer = workflow.run("高级客户售后多久响应？")

    assert answer.route == "knowledge_qa"
    assert answer.refused is False
    assert answer.review is not None
    assert answer.review.passed is True
    assert "4 小时" in answer.answer
    assert "引用来源" in answer.answer
    assert answer.citations


def test_agent_executes_whitelisted_tools(tmp_path: Path) -> None:
    store = _index_policy(tmp_path)
    workflow = AgentService(vector_store=store)

    answer = workflow.run("查询杭州某科技公司客户，并生成跟进邮件")

    assert answer.route == "tool_call"
    assert answer.refused is False
    assert [result.tool for result in answer.tool_results] == [
        "search_customer",
        "draft_followup_email",
    ]
    assert all(result.ok for result in answer.tool_results)
    assert "工具结果" in answer.answer


def test_agent_refuses_low_confidence_knowledge_question(tmp_path: Path) -> None:
    workflow = AgentService(vector_store=LocalVectorStore(index_file=tmp_path / "empty.db"))

    answer = workflow.run("完全不存在的内部制度是什么？")

    assert answer.route == "knowledge_qa"
    assert answer.refused is True
    assert answer.citations == []


def test_agent_answer_serializes_to_dict(tmp_path: Path) -> None:
    workflow = AgentService(vector_store=LocalVectorStore(index_file=tmp_path / "empty.db"))
    answer = workflow.run("查询演示客户")

    payload = agent_answer_to_dict(answer)

    assert payload["route"] == "tool_call"
    assert isinstance(payload["tool_results"], list)


class FakeLlm:
    enabled = True

    def __init__(self) -> None:
        self.messages: list[LlmMessage] = []

    def complete(self, messages: list[LlmMessage], temperature: float = 0.2) -> str:
        self.messages = messages
        return "模型答案\n\n引用来源：\n1. 文档：policy；章节：售后政策；片段：x；相似度：1.0000"


def test_agent_uses_prompt_template_and_business_resources(tmp_path: Path) -> None:
    store = _index_policy(tmp_path)
    llm = FakeLlm()
    workflow = AgentService(vector_store=store, llm=llm)  # type: ignore[arg-type]

    answer = workflow.run("高级客户售后多久响应？")

    assert answer.refused is False
    assert llm.messages
    prompt = llm.messages[-1].content
    assert "## 用户问题" in prompt
    assert "## 业务资源" in prompt
    assert "知识库问答必须基于检索证据生成" in prompt
