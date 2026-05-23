from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    path: str = Field(..., description="知识库文档路径，支持 .md、.txt、.pdf、.docx、.xlsx、.xls。")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class ConversationCreateRequest(BaseModel):
    title: str = "新对话"


class ConversationAskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AgentRunRequest(BaseModel):
    question: str = Field(..., min_length=1)


class EvaluationCaseRequest(BaseModel):
    question: str = Field(..., min_length=1)
    expected_answer_contains: str = ""
    expect_refused: bool = False
    require_citation: bool = True


class EvaluationRunRequest(BaseModel):
    cases: list[EvaluationCaseRequest] = Field(..., min_length=1)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=6)
    role: str = "user"


class UpdateUserRoleRequest(BaseModel):
    role: str = Field(..., pattern="^(user|admin)$")
