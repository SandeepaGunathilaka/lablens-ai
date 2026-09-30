from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException

from agents.explanation.models import (
    ExplanationRequest
)
from agents.explanation.service import (
    ExplanationService
)


router = APIRouter(
    prefix="/explanation",
    tags=["Explanation Agent"]
)

@lru_cache(maxsize=1)
def get_explanation_service() -> ExplanationService:
    """Initialize the model client on demand; allow offline test injection."""
    return ExplanationService()


@router.post("/explain")
def explain(
    request: ExplanationRequest,
    service: ExplanationService = Depends(get_explanation_service),
):

    try:
        explanation = service.generate_explanation(
            request
        )

        return {
            "explanation": explanation
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
