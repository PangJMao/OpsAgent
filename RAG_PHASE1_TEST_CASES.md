# RAG 第一阶段测试用例

## 测试目标

验证第一阶段 RAG 基线是否满足以下能力：

- 文档归一化为 Markdown。
- Markdown 按标题层级切分。
- 长章节或无标题内容使用重叠窗口兜底。
- chunk 元数据包含标题路径、切分策略和来源信息。
- 向量入库后可以检索并回答问题。
- 低置信度问题会拒答。
- 回答正文对普通用户展示引用来源。
- 每次入库和问答生成 trace。

## 测试前置条件

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m compileall src tests
```

如已安装开发依赖，可运行：

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m pytest
```

## 用例 1：Markdown 标题切分

**输入文档**

```markdown
# 售后政策

## 高级客户

高级客户的售后响应时间为 4 小时内。
```

**操作**

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m ops_agent.cli ingest examples\company_policy.md
```

**预期结果**

- 入库成功返回 `trace_id`。
- 返回 `chunk_count`。
- `strategy_counts` 包含 `markdown_heading`。
- chunk metadata 包含 `heading_path` 和 `chunk_strategy`。

## 用例 2：用户可见引用来源

**操作**

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m ops_agent.cli ask "高级客户售后多久响应？"
```

**预期结果**

- `refused` 为 `false`。
- `answer` 中包含 `4 小时`。
- `answer` 中包含 `引用来源：`。
- 引用来源展示文档名、章节、片段 ID 和相关度。
- `citations` 字段包含 `title`、`chunk_id`、`score`、`heading_path`。

## 用例 3：低置信度拒答

**操作**

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m ops_agent.cli ask "火星基地采购流程是什么？"
```

**预期结果**

- `refused` 为 `true`。
- `citations` 为空。
- 回答提示当前知识库没有足够依据。
- `confidence` 低于配置阈值。

## 用例 4：纯文本归一化

**输入文档**

```text
高级客户的售后响应时间为 4 小时内。
```

**预期结果**

- `.txt` 被包装为 Markdown。
- 归一化内容以 `# 文件名` 开头。
- 切分策略为 `markdown_heading`。

## 用例 5：长章节窗口兜底

**输入文档**

```markdown
# 售后政策

## 高级客户

高级客户需要 4 小时内响应。高级客户需要 4 小时内响应。……
```

**预期结果**

- 一级标题章节可按 `markdown_heading` 切分。
- 超长二级章节切成多个 chunk。
- 超长章节 chunk 的 `chunk_strategy` 为 `markdown_heading_window_fallback`。
- `fallback_used` 为 `true`。

## 用例 6：重复入库替换旧 chunk

**操作**

连续两次导入同一份文档。

**预期结果**

- 第二次入库不会让同一 `document_id` 的 chunk 数量翻倍。
- 旧 chunk 会被替换。
- 检索结果不会混入同一文档的历史切分版本。

## 用例 7：trace 生成

**操作**

执行任意一次入库或问答。

**预期结果**

- 返回 `trace_id`。
- 生成 trace 记录。
- trace 事件包含节点名、耗时、输入摘要、输出摘要。
- 入库 trace 包含切分策略分布。

## 当前已知限制

- 当前仅支持 `.md` 和 `.txt`。
- embedding 仍是本地 hashing 基线。
- SQLite 当前用于本地向量索引，正式生产路线建议切换到 PostgreSQL + pgvector。
- trace 仍是 JSON 文件，长期运行建议改为数据库 trace 表。
