from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    path: str = Field(..., description="知识库文档路径，支持 .md、.txt、.pdf、.docx、.xlsx、.xls。")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AgentRunRequest(BaseModel):
    question: str = Field(..., min_length=1)
