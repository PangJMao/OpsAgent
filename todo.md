替换 Hashing Embedding 为真实 embedding 模型，提高检索质量。
将 Agent 编排升级为 LangGraph 或更明确的状态机，增强节点可控性。

pgvector 检索只有基础相似度排序，可增加 rerank、关键词混合检索、metadata filter。

当前任务队列是内存实现，后续应接 Redis + RQ/Celery。
Review Agent 目前是规则审核，后续可接入模型审核和更严格引用校验。
工具调用目前是模拟数据，后续需要接真实 CRM、工单、邮件系统。
文档中文内容在部分源码/文档中存在编码显示异常，需要统一 UTF-8 清理。
README/启动文档内容有乱码显示问题，需要修复后用于对外展示。
前端是原生实现，后续如继续扩展，建议迁移到 React/Next.js 或至少组件化拆分。
权限目前以角色为主，知识库 scope 还比较粗，需要补充按知识库/部门/文档级授权。

缺少 Docker Compose 一键启动，目前数据库启动文档是手工命令。
缺少线上部署配置、日志聚合、OpenTelemetry、限流和审计日志。
DeepSeek API 异常目前有降级，但缺少更完整的超时、重试、熔断和成本统计。