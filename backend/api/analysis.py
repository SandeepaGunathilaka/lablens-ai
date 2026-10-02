"""Authenticated HTTP entry point for the report-analysis Coordinator workflow."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from pymongo.collection import Collection
from starlette.concurrency import run_in_threadpool

from coordinator import CoordinatorResult, analyze_report
from database import get_audit_logs_collection
from security.dependencies import get_current_user

router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/analyze-report", response_model=CoordinatorResult)
async def analyze_report_upload(
    file: Annotated[UploadFile, File(description="CBC or Lipid Profile report")],
    current_user: str = Depends(get_current_user),
    audit_logs: Collection = Depends(get_audit_logs_collection),
) -> CoordinatorResult:
    """Analyze one authenticated user's report through the Coordinator pipeline.

    The Coordinator calls the Document Agent directly, then passes its structured
    output through retrieval, explanation, and safety validation. It is synchronous
    because PDF parsing and OCR are CPU/blocking operations, so it runs in FastAPI's
    worker thread pool rather than blocking the event loop.
    """
    content = await file.read()
    return await run_in_threadpool(
        analyze_report,
        file.filename or "",
        content,
        current_user,
        audit_logs=audit_logs,
    )
