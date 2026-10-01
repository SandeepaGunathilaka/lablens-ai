"""Saved chats about a report: create, list, continue, rename, delete.

Answers come from chat_service.answer_question, so every assistant message is either
Safety-approved or the fixed fallback text. Another user's chat id behaves like an
unknown id (404).
"""

import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field, field_validator
from pymongo.collection import Collection
from starlette.concurrency import run_in_threadpool

from api.reports import find_report, stored_sources
from chat_service import ChatAnswer, answer_question
from coordinator import AnalyzedLabResult
from database import get_audit_logs_collection, get_chats_collection, get_reports_collection
from explanation_agent.router import enforce_rate_limit
from security.dependencies import get_current_user

router = APIRouter(prefix="/api/chats", tags=["chats"])

DEFAULT_TITLE = "New chat"
TITLE_LENGTH = 60


class ChatMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    created_at: datetime
    text: str | None = None
    test_names: list[str] = Field(default_factory=list)
    answer: ChatAnswer | None = None


class ChatSummary(BaseModel):
    id: str
    report_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ChatDetail(ChatSummary):
    messages: list[ChatMessage]


class CreateChatRequest(BaseModel):
    report_id: str = Field(min_length=1, max_length=128)


class RenameChatRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Title cannot be blank")
        return value


class SendMessageRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    # Limit the answer to these tests; empty means every test in the report.
    test_names: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank")
        return value


class SendMessageResponse(BaseModel):
    chat: ChatSummary
    user_message: ChatMessage
    assistant_message: ChatMessage


def _summary(doc: dict) -> ChatSummary:
    return ChatSummary(
        id=doc["chat_id"],
        report_id=doc["report_id"],
        title=doc["title"],
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        message_count=len(doc.get("messages", [])),
    )


def _detail(doc: dict) -> ChatDetail:
    return ChatDetail(**_summary(doc).model_dump(), messages=doc.get("messages", []))


def _find_chat(chats: Collection, chat_id: str, user_id: str) -> dict:
    doc = chats.find_one({"chat_id": chat_id, "user_id": user_id}, {"_id": 0})
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    return doc


@router.get("", response_model=list[ChatSummary])
def list_chats(
    report_id: str | None = Query(default=None),
    current_user: str = Depends(get_current_user),
    chats: Collection = Depends(get_chats_collection),
) -> list[ChatSummary]:
    query = {"user_id": current_user}
    if report_id:
        query["report_id"] = report_id
    return [_summary(d) for d in chats.find(query, {"_id": 0}).sort("updated_at", -1)]


@router.post("", response_model=ChatDetail, status_code=status.HTTP_201_CREATED)
def create_chat(
    body: CreateChatRequest,
    current_user: str = Depends(get_current_user),
    reports: Collection = Depends(get_reports_collection),
    chats: Collection = Depends(get_chats_collection),
) -> ChatDetail:
    find_report(reports, body.report_id, current_user)
    now = datetime.now(timezone.utc)
    doc = {
        "chat_id": str(uuid.uuid4()),
        "user_id": current_user,
        "report_id": body.report_id,
        "title": DEFAULT_TITLE,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    chats.insert_one(dict(doc))
    return _detail(doc)


@router.get("/{chat_id}", response_model=ChatDetail)
def get_chat(
    chat_id: str,
    current_user: str = Depends(get_current_user),
    chats: Collection = Depends(get_chats_collection),
) -> ChatDetail:
    return _detail(_find_chat(chats, chat_id, current_user))


@router.patch("/{chat_id}", response_model=ChatSummary)
def rename_chat(
    chat_id: str,
    body: RenameChatRequest,
    current_user: str = Depends(get_current_user),
    chats: Collection = Depends(get_chats_collection),
) -> ChatSummary:
    _find_chat(chats, chat_id, current_user)
    chats.update_one({"chat_id": chat_id, "user_id": current_user}, {"$set": {"title": body.title}})
    return _summary(_find_chat(chats, chat_id, current_user))


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat(
    chat_id: str,
    current_user: str = Depends(get_current_user),
    chats: Collection = Depends(get_chats_collection),
) -> Response:
    _find_chat(chats, chat_id, current_user)
    chats.delete_one({"chat_id": chat_id, "user_id": current_user})
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{chat_id}/messages", response_model=SendMessageResponse)
async def send_message(
    chat_id: str,
    body: SendMessageRequest,
    current_user: str = Depends(get_current_user),
    reports: Collection = Depends(get_reports_collection),
    chats: Collection = Depends(get_chats_collection),
    audit_logs: Collection = Depends(get_audit_logs_collection),
) -> SendMessageResponse:
    chat = _find_chat(chats, chat_id, current_user)
    report = find_report(reports, chat["report_id"], current_user)
    enforce_rate_limit(current_user)

    results = [AnalyzedLabResult.model_validate(r) for r in report.get("results", [])]
    wanted = {name.lower() for name in body.test_names}
    selected = [r for r in results if r.test.lower() in wanted] or results
    if not selected:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="This report has no lab values.")

    user_message = ChatMessage(
        id=str(uuid.uuid4()),
        role="user",
        created_at=datetime.now(timezone.utc),
        text=body.question,
        test_names=[r.test for r in selected],
    )
    answer = await run_in_threadpool(
        lambda: answer_question(
            report_id=report["report_id"],
            user_id=current_user,
            results=selected,
            retrieved_sources=stored_sources(report),
            question=body.question,
            audit_logs=audit_logs,
        )
    )
    assistant_message = ChatMessage(
        id=str(uuid.uuid4()), role="assistant", created_at=datetime.now(timezone.utc), answer=answer
    )

    update: dict = {
        "$push": {"messages": {"$each": [user_message.model_dump(), assistant_message.model_dump()]}},
        "$set": {"updated_at": assistant_message.created_at},
    }
    if chat["title"] == DEFAULT_TITLE and not chat.get("messages"):
        update["$set"]["title"] = body.question[:TITLE_LENGTH]
    chats.update_one({"chat_id": chat_id, "user_id": current_user}, update)

    return SendMessageResponse(
        chat=_summary(_find_chat(chats, chat_id, current_user)),
        user_message=user_message,
        assistant_message=assistant_message,
    )
