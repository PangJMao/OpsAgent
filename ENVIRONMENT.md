# 环境需求

## Python 版本

本项目要求使用 Python 3.13。

## 安装依赖

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

文档解析依赖：

- `.pdf`：`pypdf`
- `.xls`：`xlrd`
- `.docx`、`.xlsx`：当前使用标准库读取 Office Open XML 基础文本内容

## DeepSeek 配置

在项目根目录创建 `.env`，参考 `.env.example`：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
OPS_AGENT_LLM_PROVIDER=deepseek
OPS_AGENT_LLM_TIMEOUT_SECONDS=20
OPS_AGENT_MAX_RETRIES=2
```

`.env` 已被 `.gitignore` 忽略，不要提交真实 API key。

## 验证

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pytest
```
