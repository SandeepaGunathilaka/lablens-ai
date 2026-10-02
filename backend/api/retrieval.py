"""HTTP adapter for curated evidence retrieval; no patient interpretation."""

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status

from agents.retrieval_agent import MedicalRetrievalAgent
from agents.retrieval_models import RetrievalRequest, RetrievalResponse
from security.dependencies import get_current_user


router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


@lru_cache(maxsize=1)
def get_retrieval_agent() -> MedicalRetrievalAgent:
    """Reuse the agent; semantic model construction remains lazy on keyword miss."""
    return MedicalRetrievalAgent()


@router.post("", response_model=RetrievalResponse)
def retrieve(
    request: RetrievalRequest,
    current_user: str = Depends(get_current_user),
    agent: MedicalRetrievalAgent = Depends(get_retrieval_agent),
) -> RetrievalResponse:
    """Return evidence or abstentions; failures use the app's generic HTTP 500."""
    if request.user_id != current_user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed for this user")
    return agent.retrieve(request)
