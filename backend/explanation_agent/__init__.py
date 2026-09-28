"""Explanation Agent public contract.

The Coordinator should:
- call ``calculate_status`` and pass the result in ``ExplanationRequest.status``
- pass Retrieval Agent citations as ``retrieved_sources`` when they exist
- join the four narrative fields with ``build_draft_response``, which appends the
  standard disclaimer once, before sending that string to the Safety Agent
"""

from explanation_agent.copy import STANDARD_DISCLAIMER
from explanation_agent.draft import build_draft_response
from explanation_agent.models import (
    ExplanationRequest,
    ExplanationResponse,
    RetrievedSource,
)
from explanation_agent.status import StatusAssessment, calculate_status

__all__ = [
    "STANDARD_DISCLAIMER",
    "ExplanationRequest",
    "ExplanationResponse",
    "RetrievedSource",
    "StatusAssessment",
    "build_draft_response",
    "calculate_status",
]
