from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from ops_agent.models import AskRequest, IngestRequest
from ops_agent.services import RagService, answer_to_dict

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ingest")
def ingest_document(request: IngestRequest) -> dict[str, object]:
    service = RagService()
    return service.ingest(Path(request.path))


@router.post("/ask")
def ask_question(request: AskRequest) -> dict[str, object]:
    service = RagService()
    return answer_to_dict(service.ask(request.question))
