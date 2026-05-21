# Prompt 模板管理

第二阶段 Agent 的提示词集中放在 `src/ops_agent/prompts`，业务资源集中放在 `src/ops_agent/resources`：

- `prompts/templates/agent_answer.md`：模型回答模板。
- `prompts/manager.py`：读取并渲染模板。
- `resources/agent_business_rules.md`：可注入的业务规则资源。
- `resources/loader.py`：读取业务资源，并限制最大注入长度。

当前请求链路：

```text
用户输入自然语言问题
  -> 路由层接收 HTTP 请求
  -> Pydantic 校验请求参数
  -> service 层接收结构化请求
  -> AgentService 检索知识库并执行白名单工具
  -> 读取 Prompt 模板
  -> 读取 resources 中的业务资源
  -> 组装大模型上下文
  -> 调用 LLM 或本地降级生成
  -> 获得模型输出
  -> service 层解析、校验、格式化
  -> 返回结构化响应
```

新增提示词时优先新增模板文件，不要在业务代码中散落大段 prompt 字符串。业务资源应保持短小，并由 `ResourceLoader` 控制注入长度，避免上下文持续膨胀。
