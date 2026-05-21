from pathlib import Path

from ops_agent.prompts import PromptManager, PromptRenderInput, ResourceLoader


def test_prompt_manager_renders_template(tmp_path: Path) -> None:
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "test.md").write_text("Q={question}\nR={business_resources}", encoding="utf-8")

    prompt = PromptManager(template_dir=template_dir).render(
        "test.md",
        PromptRenderInput(
            question="客户问题",
            route="knowledge_qa",
            evidence="证据",
            tool_results="工具结果",
            business_resources="业务规则",
        ),
    )

    assert "Q=客户问题" in prompt
    assert "R=业务规则" in prompt


def test_resource_loader_truncates_long_resources(tmp_path: Path) -> None:
    resource_dir = tmp_path / "resources"
    resource_dir.mkdir()
    (resource_dir / "rules.md").write_text("abcdef", encoding="utf-8")

    content = ResourceLoader(resource_dir=resource_dir, max_chars=3).load("rules.md")

    assert content == "abc\n\n[resource truncated]"
