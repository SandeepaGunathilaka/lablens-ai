from typing import Optional

from pydantic import BaseModel


class MedicalFinding(BaseModel):
    test_name: str
    value: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    status: Optional[str] = None


class ExplanationRequest(BaseModel):
    findings: list[MedicalFinding]
    patient_context: Optional[str] = None


class ExplainedFinding(BaseModel):
    test_name: str
    result: str
    status: Optional[str] = None
    what_it_measures: str
    explanation: str
    possible_meaning: str
    recommended_discussion: str


class SafetyInformation(BaseModel):
    is_diagnostic: bool
    requires_professional_review: bool


class ExplanationResponse(BaseModel):
    findings: list[ExplainedFinding]
    safety: SafetyInformation