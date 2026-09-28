"""HTTP endpoint for the Explanation Agent.

Input size limits are enforced by the request model. Authorization should be
added with the shared JWT middleware when that foundation is wired in.
"""

import os
import threading
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from explanation_agent.models import ExplanationRequest, ExplanationResponse
from explanation_agent.service import ExplanationService, build_explanation_service

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


def enforce_rate_limit(request: Request) -> None:
    raw_limit = os.getenv("EXPLANATION_RATE_LIMIT", "30").strip()
    try:
        limit = int(raw_limit)
    except ValueError:
        limit = 30
    if limit <= 0:
        return

    host = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window_start = now - 60
    with _lock:
        recent = [stamp for stamp in _hits.get(host, []) if stamp >= window_start]
        if len(recent) >= limit:
            _hits[host] = recent
            raise HTTPException(
                status_code=429,
                detail="Too many explanation requests. Try again shortly.",
            )
        recent.append(now)
        _hits[host] = recent


@router.post("/explanation", response_model=ExplanationResponse)
def create_explanation(
    body: ExplanationRequest,
    request: Request,
    service: ExplanationService = Depends(get_explanation_service),
) -> ExplanationResponse:
    enforce_rate_limit(request)
    return service.explain(body)
