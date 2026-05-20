from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ops_agent.rag import RagPipeline, answer_to_dict

app = FastAPI(title="OpsAgent API", version="0.1.0")


class IngestRequest(BaseModel):
    path: str = Field(..., description="`.md` 或 `.txt` 文档的绝对路径或项目相对路径。")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rag/ingest")
def ingest_document(request: IngestRequest) -> dict[str, object]:
    pipeline = RagPipeline()
    return pipeline.ingest(Path(request.path))


@app.post("/rag/ask")
def ask_question(request: AskRequest) -> dict[str, object]:
    pipeline = RagPipeline()
    return answer_to_dict(pipeline.ask(request.question))
