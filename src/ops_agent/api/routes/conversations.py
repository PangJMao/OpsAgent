from __future__ import annotations

import json
import time
from typing import Any, Iterable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ops_agent.api.security import require_login
from ops_agent.models import ConversationAskRequest, ConversationCreateRequest
from ops_agent.services import RagService, answer_to_dict
from ops_agent.services.conversation_service import conversation_service
from ops_agent.services.database_service import StartupConfigurationError

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("")
def list_conversations(request: Request) -> dict[str, object]:
    user = require_login(request)
    try:
        return {"conversations": conversation_service.list_conversations(user.user_id)}
    except StartupConfigurationError as exc:
        raise HTTPException(status_code=503, detail=f"对话存储不可用：{exc}") from exc


@router.post("")
def create_conversation(request: Request, payload: ConversationCreateRequest) -> dict[str, object]:
    user = require_login(request)
    try:
        return {"conversation": conversation_service.create_conversation(user.user_id, payload.title)}
    except StartupConfigurationError as exc:
        raise HTTPException(status_code=503, detail=f"对话存储不可用：{exc}") from exc


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str, request: Request) -> dict[str, object]:
    user = require_login(request)
    try:
        conversation_service.delete_conversation(user.user_id, conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found.") from exc
    except StartupConfigurationError as exc:
        raise HTTPException(status_code=503, detail=f"对话存储不可用：{exc}") from exc
    return {"ok": True}


@router.get("/{conversation_id}/messages")
def list_messages(conversation_id: str, request: Request) -> dict[str, object]:
    user = require_login(request)
    try:
        return {"messages": conversation_service.list_messages(user.user_id, conversation_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found.") from exc
    except StartupConfigurationError as exc:
        raise HTTPException(status_code=503, detail=f"对话存储不可用：{exc}") from exc


@router.post("/{conversation_id}/messages")
def ask_in_conversation(
    conversation_id: str,
    request: Request,
    payload: ConversationAskRequest,
) -> dict[str, object]:
    user = require_login(request)
    try:
        conversation_service.add_message(user.user_id, conversation_id, "user", payload.question)
        answer = RagService().ask(payload.question)
        answer_payload = answer_to_dict(answer)
        conversation_service.add_message(
            user.user_id,
            conversation_id,
            "assistant",
            answer.answer,
        citations=_public_citations(answer.citations),
        )
        return {"answer": answer_payload, "messages": conversation_service.list_messages(user.user_id, conversation_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found.") from exc
    except (RuntimeError, ValueError, StartupConfigurationError) as exc:
        raise HTTPException(status_code=503, detail=f"对话处理失败：{exc}") from exc


@router.post("/{conversation_id}/messages/stream")
def stream_conversation_answer(
    conversation_id: str,
    request: Request,
    payload: ConversationAskRequest,
) -> StreamingResponse:
    user = require_login(request)

    def events() -> Iterable[str]:
        try:
            service = RagService()
            conversation_service.add_message(user.user_id, conversation_id, "user", payload.question)
            answer = service.ask(payload.question)
            citations = _public_citations(answer.citations)
            reasoning = service.build_public_reasoning(payload.question, answer)

            yield from _yield_text("thought", reasoning)
            yield from _yield_text("answer", answer.answer)

            conversation_service.add_message(
                user.user_id,
                conversation_id,
                "assistant",
                answer.answer,
                citations=citations,
            )
            messages = conversation_service.list_messages(user.user_id, conversation_id)
            yield _sse("citations", {"citations": citations})
            yield _sse("done", {"answer": answer_to_dict(answer), "messages": messages})
        except KeyError:
            yield _sse("error", {"detail": "Conversation not found."})
        except (RuntimeError, StartupConfigurationError, ValueError) as exc:
            yield _sse("error", {"detail": f"对话处理失败：{exc}"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _yield_text(event: str, text: str) -> Iterable[str]:
    for chunk in _chunk_text(text):
        yield _sse(event, {"delta": chunk})
        time.sleep(0.01)


def _chunk_text(text: str, size: int = 8) -> Iterable[str]:
    buffer = ""
    for char in text:
        buffer += char
        if len(buffer) >= size or char in "\n。；，、.!?？":
            yield buffer
            buffer = ""
    if buffer:
        yield buffer


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _public_citations(citations) -> list[dict[str, object]]:
    return [
        {
            "title": citation.title,
            "heading_path": citation.heading_path,
        }
        for citation in citations
    ]
