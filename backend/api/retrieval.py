"""HTTP adapter for curated evidence retrieval; no patient interpretation."""

from functools import lru_cache

from fastapi import APIRouter, Depends

from agents.retrieval_agent import MedicalRetrievalAgent
from agents.retrieval_models import RetrievalRequest, RetrievalResponse


router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


@lru_cache(maxsize=1)
def get_retrieval_agent() -> MedicalRetrievalAgent:
    """Reuse the agent; semantic model construction remains lazy on keyword miss."""
    return MedicalRetrievalAgent()


@router.post("", response_model=RetrievalResponse)
def retrieve(
    request: RetrievalRequest,
    agent: MedicalRetrievalAgent = Depends(get_retrieval_agent),
) -> RetrievalResponse:
    """Return evidence or abstentions; failures use the app's generic HTTP 500."""
    return agent.retrieve(request)
