"""HTTP endpoint for the Explanation Agent. Input size limits live on the request model."""

import os
import threading
import time
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.collection import Collection

from database import get_audit_logs_collection
from explanation_agent.models import ExplanationRequest, ExplanationResponse
from explanation_agent.service import ExplanationService, build_explanation_service
from logging_service import log_event
from security.dependencies import get_current_user

router = APIRouter(tags=["explanation"])

_service: ExplanationService | None = None
_hits: dict[str, list[float]] = {}
_lock = threading.Lock()


def get_explanation_service() -> ExplanationService:
    global _service
    if _service is None:
        _service = build_explanation_service()
    return _service


def reset_explanation_service() -> None:
    global _service
    _service = None


def clear_rate_limits() -> None:
    with _lock:
        _hits.clear()


def enforce_rate_limit(user_id: str) -> None:
    raw_limit = os.getenv("EXPLANATION_RATE_LIMIT", "30").strip()
    try:
        limit = int(raw_limit)
    except ValueError:
        limit = 30
    if limit <= 0:
        return

    now = time.monotonic()
    window_start = now - 60
    with _lock:
        recent = [stamp for stamp in _hits.get(user_id, []) if stamp >= window_start]
        if len(recent) >= limit:
            _hits[user_id] = recent
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many explanation requests. Try again shortly.",
            )
        recent.append(now)
        _hits[user_id] = recent


@router.post("/explanation", response_model=ExplanationResponse)
def create_explanation(
    body: ExplanationRequest,
    service: ExplanationService = Depends(get_explanation_service),
    audit_logs: Collection = Depends(get_audit_logs_collection),
    current_user: str = Depends(get_current_user),
) -> ExplanationResponse:
    # Checked before anything runs or is logged, same as the Safety Agent.
    if body.user_id != current_user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed for this user")
    enforce_rate_limit(current_user)

    response = service.explain(body)

    # Counts only: no values, test names, or explanation text.
    modes = Counter(finding.generation_mode for finding in response.findings)
    log_event(
        audit_logs,
        task_id=body.task_id,
        report_id=body.report_id,
        user_id=body.user_id,
        agent="explanation_agent",
        action="explain",
        status="success",
        details={"finding_count": len(response.findings), "generation_modes": dict(modes)},
    )
    return response
