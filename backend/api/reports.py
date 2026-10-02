"""Saved reports: upload-and-analyze, list, rename, delete, and the original file.

Every query is filtered by the caller's user_id; another user's report id behaves
exactly like an unknown id (404), so the response never reveals that it exists.
"""

import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import PurePath
from typing import Annotated, Literal

from bson import Binary
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, Field, field_validator
from pymongo.collection import Collection
from starlette.concurrency import run_in_threadpool

from agents.retrieval_models import RetrievalRequest
from agents.safety_agent import RetrievedSource
from chat_service import SourceLink, source_links
from coordinator import AnalyzedLabResult, analyze_report, default_retrieval, retrieval_response_to_sources
from database import (
    get_audit_logs_collection,
    get_chats_collection,
    get_report_files_collection,
    get_reports_collection,
)
from security.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reports", tags=["reports"])

CONTENT_TYPES = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


class SourceInfo(BaseModel):
    test_name: str
    passages: list[str] = Field(default_factory=list)
    links: list[SourceLink] = Field(default_factory=list)


class ReportSummary(BaseModel):
    id: str
    task_id: str
    name: str
    report_type: str | None = None
    status: Literal["approved", "fallback"]
    created_at: datetime
    has_file: bool
    test_count: int
    result_statuses: list[Literal["low", "normal", "high"] | None]
    chat_count: int = 0


class ReportDetail(ReportSummary):
    original_filename: str
    content_type: str | None = None
    results: list[AnalyzedLabResult]
    final_response: str | None = None
    message: str | None = None
    sources: list[SourceInfo] = Field(default_factory=list)


class RenameReportRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Name cannot be blank")
        return value


def ensure_report_indexes(reports: Collection, report_files: Collection, chats: Collection) -> None:
    reports.create_index("report_id", unique=True)
    reports.create_index([("user_id", 1), ("created_at", -1)])
    report_files.create_index("report_id", unique=True)
    chats.create_index("chat_id", unique=True)
    chats.create_index([("user_id", 1), ("report_id", 1), ("updated_at", -1)])


def find_report(reports: Collection, report_id: str, user_id: str) -> dict:
    doc = reports.find_one({"report_id": report_id, "user_id": user_id}, {"_id": 0})
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return doc


def stored_sources(doc: dict) -> list[RetrievedSource]:
    return [RetrievedSource.model_validate(s) for s in doc.get("sources", [])]


def _summary_fields(doc: dict, chat_count: int) -> dict:
    results = doc.get("results", [])
    return {
        "id": doc["report_id"],
        "task_id": doc["task_id"],
        "name": doc["name"],
        "report_type": doc.get("report_type"),
        "status": doc["status"],
        "created_at": doc["created_at"],
        "has_file": doc.get("has_file", False),
        "test_count": len(results),
        "result_statuses": [r.get("status") for r in results],
        "chat_count": chat_count,
    }


def to_detail(doc: dict, chat_count: int) -> ReportDetail:
    sources = []
    for source in stored_sources(doc):
        passages = source.information.get("passages", [])
        sources.append(
            SourceInfo(
                test_name=source.test_name,
                passages=[str(p) for p in passages] if isinstance(passages, list) else [],
                links=source_links(source),
            )
        )
    return ReportDetail(
        **_summary_fields(doc, chat_count),
        original_filename=doc.get("original_filename", ""),
        content_type=doc.get("content_type"),
        results=doc.get("results", []),
        final_response=doc.get("final_response"),
        message=doc.get("message"),
        sources=sources,
    )


def _retrieve_sources(task_id: str, report_id: str, user_id: str, results: list[AnalyzedLabResult]) -> list[dict]:
    """Curated evidence kept with the report, so chat answers use the same sources."""
    if not results:
        return []
    try:
        request = RetrievalRequest(
            task_id=task_id, report_id=report_id, user_id=user_id, test_names=[r.test for r in results]
        )
        return [s.model_dump(mode="json") for s in retrieval_response_to_sources(default_retrieval(request))]
    except Exception:
        logger.exception("Retrieval for saved report %s failed; saving without sources", report_id)
        return []


@router.post("", response_model=ReportDetail, status_code=status.HTTP_201_CREATED)
async def upload_report(
    file: Annotated[UploadFile, File(description="CBC or Lipid Profile report")],
    current_user: str = Depends(get_current_user),
    reports: Collection = Depends(get_reports_collection),
    report_files: Collection = Depends(get_report_files_collection),
    audit_logs: Collection = Depends(get_audit_logs_collection),
) -> ReportDetail:
    """Analyze an upload and save it. A failed analysis is not saved (422 with the reason)."""
    filename = file.filename or ""
    content = await file.read()
    result = await run_in_threadpool(analyze_report, filename, content, current_user, audit_logs=audit_logs)
    if result.status == "failed":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=result.message)

    sources = await run_in_threadpool(
        _retrieve_sources, result.task_id, result.report_id, current_user, result.results
    )
    suffix = PurePath(filename).suffix.lower()
    content_type = CONTENT_TYPES.get(suffix)
    doc = {
        "report_id": result.report_id,
        "task_id": result.task_id,
        "user_id": current_user,
        "name": PurePath(filename).stem or "Lab report",
        "original_filename": filename,
        "content_type": content_type,
        "has_file": True,
        "created_at": datetime.now(timezone.utc),
        "status": result.status,
        "report_type": result.report_type,
        "results": [r.model_dump() for r in result.results],
        "final_response": result.final_response,
        "message": result.message,
        "sources": sources,
    }
    report_files.insert_one(
        {
            "report_id": result.report_id,
            "user_id": current_user,
            "filename": filename,
            "content_type": content_type,
            "content": Binary(content),
        }
    )
    reports.insert_one(dict(doc))
    return to_detail(doc, chat_count=0)


@router.get("", response_model=list[ReportSummary])
def list_reports(
    current_user: str = Depends(get_current_user),
    reports: Collection = Depends(get_reports_collection),
    chats: Collection = Depends(get_chats_collection),
) -> list[ReportSummary]:
    chat_counts = Counter(c["report_id"] for c in chats.find({"user_id": current_user}, {"report_id": 1}))
    docs = reports.find({"user_id": current_user}, {"_id": 0}).sort("created_at", -1)
    return [ReportSummary(**_summary_fields(d, chat_counts[d["report_id"]])) for d in docs]


@router.get("/{report_id}", response_model=ReportDetail)
def get_report(
    report_id: str,
    current_user: str = Depends(get_current_user),
    reports: Collection = Depends(get_reports_collection),
    chats: Collection = Depends(get_chats_collection),
) -> ReportDetail:
    doc = find_report(reports, report_id, current_user)
    return to_detail(doc, chats.count_documents({"report_id": report_id, "user_id": current_user}))


@router.patch("/{report_id}", response_model=ReportDetail)
def rename_report(
    report_id: str,
    body: RenameReportRequest,
    current_user: str = Depends(get_current_user),
    reports: Collection = Depends(get_reports_collection),
    chats: Collection = Depends(get_chats_collection),
) -> ReportDetail:
    find_report(reports, report_id, current_user)
    reports.update_one({"report_id": report_id, "user_id": current_user}, {"$set": {"name": body.name}})
    return get_report(report_id, current_user, reports, chats)


@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_report(
    report_id: str,
    current_user: str = Depends(get_current_user),
    reports: Collection = Depends(get_reports_collection),
    report_files: Collection = Depends(get_report_files_collection),
    chats: Collection = Depends(get_chats_collection),
) -> Response:
    """Deletes the report, its original file and all of its chats."""
    find_report(reports, report_id, current_user)
    owned = {"report_id": report_id, "user_id": current_user}
    chats.delete_many(owned)
    report_files.delete_many(owned)
    reports.delete_one(owned)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{report_id}/file")
def get_report_file(
    report_id: str,
    current_user: str = Depends(get_current_user),
    report_files: Collection = Depends(get_report_files_collection),
) -> Response:
    doc = report_files.find_one({"report_id": report_id, "user_id": current_user})
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return Response(
        content=bytes(doc["content"]),
        media_type=doc.get("content_type") or "application/octet-stream",
        headers={"Cache-Control": "private, no-store"},
    )


@router.delete("/{report_id}/file", response_model=ReportDetail)
def delete_report_file(
    report_id: str,
    current_user: str = Depends(get_current_user),
    reports: Collection = Depends(get_reports_collection),
    report_files: Collection = Depends(get_report_files_collection),
    chats: Collection = Depends(get_chats_collection),
) -> ReportDetail:
    """Removes only the uploaded original; the extracted results and chats stay."""
    find_report(reports, report_id, current_user)
    owned = {"report_id": report_id, "user_id": current_user}
    report_files.delete_many(owned)
    reports.update_one(owned, {"$set": {"has_file": False}})
    return get_report(report_id, current_user, reports, chats)
