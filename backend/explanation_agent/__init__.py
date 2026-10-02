"""Explanation Agent public contract.

The Coordinator should:
- pass each finding with the status it computed in code
- pass the same ``retrieved_sources`` list it sends to the Safety Agent
- on a Safety rejection, call again with the regeneration instruction in
  ``rejection_feedback``
- join the four narrative fields of each ``ExplainedFinding`` into the draft and
  append the Safety Agent's ``DISCLAIMER`` itself
"""

from explanation_agent.models import (
    ExplainedFinding,
    ExplanationFinding,
    ExplanationRequest,
    ExplanationResponse,
)
from explanation_agent.service import ExplanationService, build_explanation_service

__all__ = [
    "ExplainedFinding",
    "ExplanationFinding",
    "ExplanationRequest",
    "ExplanationResponse",
    "ExplanationService",
    "build_explanation_service",
]
