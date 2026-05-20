# RAG 最小可用版本

当前第一阶段实现一个本地可运行的 RAG 基线，重点先把文档处理边界、引用、拒答、向量入库和可观测性做稳定。

## 文档处理策略

当前采用“文档归一化为 Markdown，再按标题切分，重叠窗口兜底”的策略：

```text
原始文档
  -> 归一化为 Markdown
  -> 按 Markdown 标题层级切分
  -> 长章节或无结构文本使用重叠窗口兜底
  -> 写入向量索引
```

当前支持：

- `.md`：直接作为 Markdown 中间表示。
- `.txt`：包装为带一级标题的 Markdown。

后续接入 PDF、DOCX、HTML 时，只需要扩展 `normalizers.py`，下游切分器仍然只处理统一的 Markdown。

## 架构边界

- `src/ops_agent/ingestion/normalizers.py`：把不同格式文档归一化为 Markdown。
- `src/ops_agent/ingestion/chunker.py`：按 Markdown 标题切分，必要时使用重叠窗口兜底。
- `src/ops_agent/retrieval`：embedding 与本地 SQLite 向量索引。
- `src/ops_agent/rag`：RAG 应用编排，包括检索、置信度门控、答案组织和引用返回。
- `src/ops_agent/observability`：trace 记录，每个节点记录输入摘要、输出摘要、耗时和错误。
- `src/ops_agent/api`：FastAPI 入口，给前端或外部系统调用。

## 切分元数据

每个 chunk 会保留关键元数据：

```json
{
  "source_format": "markdown",
  "normalized_format": "markdown",
  "heading_path": ["售后政策", "高级客户"],
  "heading_level": 2,
  "chunk_strategy": "markdown_heading",
  "fallback_used": false
}
```

长章节会使用 `markdown_heading_window_fallback`，无标题内容会使用 `overlap_window_fallback`。

## 引用来源展示

回答正文会追加用户可见的引用来源区块，例如：

```text
引用来源：
1. 文档：售后政策；章节：售后政策 > 高级客户；片段：xxxx；相关度：0.3291
```

API 返回的 `citations` 字段也会包含 `document_id`、`title`、`chunk_id`、`score`、`heading_path` 和 `chunk_strategy`。

## 本地运行

安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

入库示例文档：

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m ops_agent.cli ingest examples\company_policy.md
```

提问：

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m ops_agent.cli ask "高级客户售后多久响应？"
```

启动 API：

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\uvicorn.exe ops_agent.api.main:app --reload
```

## 可观测性

每次入库或提问都会生成 `trace_id`，并把执行链路写入：

```text
storage/traces/<trace_id>.json
```

trace 中包含：

- 节点名称
- 开始时间
- 节点耗时
- 输入摘要
- 输出摘要
- 异常信息
- 入库时的切分策略分布

## 这种方法的优势

对比直接在 PDF、Word 等原始格式上切分，统一归一化为 Markdown 后再切分有几个优势：

- **职责更清楚**：解析器负责格式转换，切分器只负责 Markdown 结构切分，代码边界更稳定。
- **切分更贴近语义**：标题层级天然表达章节关系，比固定窗口更不容易切断规则、例外条件和上下文。
- **引用更清晰**：chunk 可以带上 `heading_path`，后续展示 citation 时能指向具体章节。
- **可观测性更好**：归一化结果会保存到 `storage/normalized`，方便判断问题出在转换、切分还是检索。
- **扩展成本更低**：新增 PDF、DOCX、HTML 解析时，只需要新增 normalizer，不需要改 RAG 主流程。
- **保留兜底能力**：标题缺失或章节过长时仍然使用重叠窗口，避免因为结构质量差而无法入库。

## 当前限制

- 当前只实现 `.md` 和 `.txt` 的归一化。
- embedding 是本地 hashing 基线实现，不依赖外部模型。
- 本地向量索引用 SQLite 存储 embedding 和 chunk 元数据，正式生产路线仍建议切换到 PostgreSQL + pgvector。
- 答案生成是抽取式归纳模板，后续会接入真实对话模型与审核 Agent。
