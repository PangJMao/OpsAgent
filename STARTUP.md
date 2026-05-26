# OpsAgent 手动启动文档

本文档用于手动启动 OpsAgent，包括 PostgreSQL + pgvector、数据库初始化、Python 依赖安装、后端服务和前端页面。

## 1. 环境要求

- Windows
- Python：`D:\Users\Liuhj\anaconda3\python.exe`
- Docker Desktop
- DBeaver 可选，用于查看数据库
- 项目目录：

```powershell
D:\Agent_Demo\OpsAgent
```

## 2. 启动 pgvector 数据库

本项目使用 PostgreSQL + pgvector 作为向量知识库。推荐只用 Docker 启动数据库，应用仍在本机 Python 环境运行。

确认 Docker Desktop 已启动：

```powershell
docker version
```

首次启动 pgvector 容器：

```powershell
docker run --name ops-agent-pgvector `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_PASSWORD=123456 `
  -e POSTGRES_DB=postgres `
  -p 5433:5432 `
  -d pgvector/pgvector:pg17
```

如果容器已经创建过，直接启动：

```powershell
docker start ops-agent-pgvector
```

查看状态：

```powershell
docker ps
```

## 3. 初始化项目数据库

进入数据库：

```powershell
docker exec -it ops-agent-pgvector psql -U postgres -d postgres
```

在 `postgres=#` 中执行：

```sql
CREATE USER ops_agent WITH PASSWORD 'ops_agent';
CREATE DATABASE ops_agent OWNER ops_agent;
\c ops_agent
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extname FROM pg_extension WHERE extname = 'vector';
\q
```

如果用户或数据库已存在，使用：

```sql
ALTER USER ops_agent WITH PASSWORD 'ops_agent';
ALTER DATABASE ops_agent OWNER TO ops_agent;
```

## 4. 配置 .env

确认 `D:\Agent_Demo\OpsAgent\.env` 至少包含：

```env
DEEPSEEK_API_KEY=你的模型APIKey
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com/anthropic
OPS_AGENT_LLM_PROVIDER=deepseek
OPS_AGENT_LLM_TIMEOUT_SECONDS=20
OPS_AGENT_MAX_RETRIES=2

OPS_AGENT_REQUIRE_EXTERNAL_SERVICES=true
OPS_AGENT_DATABASE_URL=postgresql://ops_agent:ops_agent@localhost:5433/ops_agent
OPS_AGENT_VECTOR_PROVIDER=pgvector
OPS_AGENT_ROOT_USERNAME=root
OPS_AGENT_ROOT_PASSWORD=123456
OPS_AGENT_SESSION_SECRET=change-me-session-secret

OPS_AGENT_RERANK_PROVIDER=bge
OPS_AGENT_RERANK_MODEL=BAAI/bge-reranker-base
OPS_AGENT_RERANK_USE_FP16=true
OPS_AGENT_RERANK_REQUIRE_MODEL=false
OPS_AGENT_RETRIEVAL_TOP_K=12
OPS_AGENT_RERANK_TOP_K=3
```

注意：如果使用上面的 Docker 数据库容器，数据库端口必须是 `5433`。

## 5. 安装依赖

```powershell
cd D:\Agent_Demo\OpsAgent
D:\Users\Liuhj\anaconda3\python.exe -m pip install -r requirements.txt
D:\Users\Liuhj\anaconda3\python.exe -m pip install -e .
```

`FlagEmbedding` 会安装 Torch、Transformers 等依赖，首次安装会比较慢。如果只是做基础演示，可保持：

```env
OPS_AGENT_RERANK_REQUIRE_MODEL=false
```

BGE 模型不可用时系统会降级到本地轻量 rerank。

## 6. 启动系统

```powershell
cd D:\Agent_Demo\OpsAgent
D:\Users\Liuhj\anaconda3\python.exe -m uvicorn ops_agent.main:app --host 127.0.0.1 --port 8000
```

启动成功后打开：

```text
http://127.0.0.1:8000/
```

默认 root 登录信息：

```text
用户名：root
密码：123456
```

## 7. 健康检查

浏览器打开：

```text
http://127.0.0.1:8000/health
```

数据库正常时应看到：

```json
"database": {
  "status": "ok"
}
```

如果看到：

```json
"status": "degraded"
```

说明系统已降级运行，查看 `startup_errors` 判断原因。

## 8. DBeaver 连接参数

连接 Docker pgvector：

```text
Host: localhost
Port: 5433
Database: ops_agent
Username: ops_agent
Password: ops_agent
```

也可以用超级用户连接：

```text
Host: localhost
Port: 5433
Database: postgres
Username: postgres
Password: 123456
```

## 9. RAG 检索与 Rerank

当前 RAG 流程：

```text
用户提问
  -> FastAPI 接口接收请求
  -> Pydantic 校验请求结构
  -> Service 层组织 Agent / RAG 流程
  -> 向量库初步召回 top 12
  -> BGE Reranker 重新判断相关性
  -> 取最相关 top 3
  -> 用户问题 + 知识片段 + Prompt 规则发送给 LLM
  -> LLM 基于上下文生成答案
  -> Service 层整理结果
  -> FastAPI 返回给调用方
```

相关参数：

```env
OPS_AGENT_RETRIEVAL_TOP_K=12
OPS_AGENT_RERANK_TOP_K=3
OPS_AGENT_RERANK_PROVIDER=bge
OPS_AGENT_RERANK_MODEL=BAAI/bge-reranker-base
OPS_AGENT_RERANK_REQUIRE_MODEL=false
```

## 10. 文档入库流程

管理员上传文档后：

```text
上传文件
  -> 保存原始文件到 storage/documents
  -> 归一化为 Markdown
  -> 保存标准 Markdown 到 storage/normalized
  -> 按标题或窗口切分 chunk
  -> 生成 embedding
  -> 写入 PostgreSQL + pgvector 的 knowledge_chunks 表
```

支持格式：

```text
.md
.txt
.pdf
.docx
.xlsx
.xls
```

## 11. 常见问题

### Docker 提示本地没有镜像

这是正常的，Docker 会自动下载：

```text
Unable to find image 'pgvector/pgvector:pg17' locally
```

等待下载完成即可。

### 端口 5433 被占用

改用其他端口，例如 `5434`：

```powershell
docker run --name ops-agent-pgvector `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_PASSWORD=123456 `
  -e POSTGRES_DB=postgres `
  -p 5434:5432 `
  -d pgvector/pgvector:pg17
```

同时修改 `.env`：

```env
OPS_AGENT_DATABASE_URL=postgresql://ops_agent:ops_agent@localhost:5434/ops_agent
```

### pgvector 不可用

如果执行：

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

报错：

```text
extension "vector" is not available
```

说明你连接的不是 Docker pgvector 容器，或者使用了本机 PostgreSQL。确认端口是否是 `5433`。

### 系统 degraded

打开：

```text
http://127.0.0.1:8000/health
```

查看：

```json
startup_errors
```

常见原因：

- PostgreSQL 容器没启动
- `.env` 端口配置错误
- `ops_agent` 数据库不存在
- `ops_agent` 用户密码错误
- `vector` 扩展没有启用
- `OPS_AGENT_RERANK_REQUIRE_MODEL=true` 但 BGE 模型无法加载

### 重新创建数据库容器

会删除容器内数据库数据：

```powershell
docker stop ops-agent-pgvector
docker rm ops-agent-pgvector
```

然后重新执行第 2、3 步。
