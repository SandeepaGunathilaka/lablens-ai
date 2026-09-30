from fastapi import APIRouter, HTTPException

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

service = ExplanationService()


@router.post("/explain")
def explain(request: ExplanationRequest):

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