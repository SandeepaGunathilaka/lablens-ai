"""Answers a follow-up question about a saved report.

Same safety contract as the Coordinator: Explanation -> draft -> Safety, regenerating up
to MAX_EXPLANATION_RETRIES times on rejection. Only the fields of a Safety-approved draft
are returned; otherwise the caller gets the fallback message and no generated text.
"""

import logging
import uuid
from collections import Counter
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field
from pymongo.collection import Collection

from agents.safety_agent import (
    ApprovedResponse,
    LabResult,
    RejectedResponse,
    RetrievedSource,
    SafetyValidateRequest,
    validate_draft,
)
from coordinator import (
    EXPLANATION_UNAVAILABLE_MESSAGE,
    FALLBACK_MESSAGE,
    INSUFFICIENT_INFORMATION_MESSAGE,
    MAX_EXPLANATION_RETRIES,
    AnalyzedLabResult,
    _match_explained,
    build_draft_response,
    default_explanation,
    rejection_note,
)
from chat_intent import ChatIntent
from explanation_agent.models import ExplanationFinding, ExplanationRequest, ExplanationResponse, GenerationMode
from logging_service import log_event

logger = logging.getLogger(__name__)

ExplanationFn = Callable[[ExplanationRequest], ExplanationResponse]
SafetyService = Callable[[SafetyValidateRequest], ApprovedResponse | RejectedResponse]


class SourceLink(BaseModel):
    title: str
    url: str | None = None


class AnswerFinding(BaseModel):
    test: str
    value: float
    unit: str | None = None
    reference_range: str | None = None
    status: Literal["low", "normal", "high"] | None = None
    what_it_measures: str = ""
    explanation: str = ""
    possible_meaning: str = ""
    recommended_discussion: str = ""
    insufficient_information: bool = False
    generation_mode: GenerationMode | None = None
    sources: list[SourceLink] = Field(default_factory=list)


class ChatAnswer(BaseModel):
    task_id: str
    # "reply" and "redirect" are answered from the question alone, without generated text.
    status: Literal["approved", "fallback", "reply", "redirect"]
    findings: list[AnswerFinding] = Field(default_factory=list)
    message: str | None = None


def intent_reply(*, report_id: str, user_id: str, intent: ChatIntent, audit_logs: Collection) -> ChatAnswer:
    """Answer a question that needs no explanation: a doctor referral, small talk or an off-topic note."""
    task_id = str(uuid.uuid4())
    status = "redirect" if intent.kind == "diagnosis" else "reply"
    log_event(
        audit_logs,
        task_id=task_id,
        report_id=report_id,
        user_id=user_id,
        agent="coordinator",
        action="answer_question",
        status=status,
        details={"intent": intent.kind},
    )
    return ChatAnswer(task_id=task_id, status=status, message=intent.reply)


def source_links(source: RetrievedSource | None) -> list[SourceLink]:
    if source is None:
        return []
    links = []
    for item in source.sources:
        if isinstance(item, dict) and item.get("title"):
            links.append(SourceLink(title=str(item["title"]), url=item.get("url")))
    return links


def answer_question(
    *,
    report_id: str,
    user_id: str,
    results: list[AnalyzedLabResult],
    retrieved_sources: list[RetrievedSource],
    question: str,
    audit_logs: Collection,
    explanation_service: ExplanationFn | None = None,
    safety_service: SafetyService | None = None,
) -> ChatAnswer:
    """Explain `results` in light of `question`. Never raises for agent failures."""
    explanation_service = explanation_service or default_explanation
    safety_service = safety_service or validate_draft
    task_id = str(uuid.uuid4())

    def audit(agent: str, action: str, status: str, **details) -> None:
        # Ids, outcomes and counts only: never the question, values or answer text.
        log_event(
            audit_logs,
            task_id=task_id,
            report_id=report_id,
            user_id=user_id,
            agent=agent,
            action=action,
            status=status,
            details=details,
        )

    def fallback(message: str) -> ChatAnswer:
        audit("coordinator", "answer_question", "rejected", outcome="fallback")
        return ChatAnswer(task_id=task_id, status="fallback", message=message)

    by_test = {s.test_name.lower(): s for s in retrieved_sources}
    tests_with_sources = {r.test.lower() for r in results if r.test.lower() in by_test}
    relevant_sources = [by_test[name] for name in tests_with_sources]

    rejection_feedback: list[str] = []
    for attempt in range(1, MAX_EXPLANATION_RETRIES + 2):
        try:
            request = ExplanationRequest(
                task_id=task_id,
                report_id=report_id,
                user_id=user_id,
                findings=[
                    ExplanationFinding(
                        test=r.test, value=r.value, unit=r.unit, reference_range=r.reference_range, status=r.status
                    )
                    for r in results
                ],
                retrieved_sources=relevant_sources,
                rejection_feedback=list(rejection_feedback),
                user_question=question,
            )
            explanation = ExplanationResponse.model_validate(explanation_service(request))
            explained = explanation.findings
            audit(
                "explanation_agent",
                "explain",
                "success",
                attempt=attempt,
                generation_modes=dict(sorted(Counter(f.generation_mode for f in explained).items())),
            )
        except Exception as exc:
            logger.exception("Explanation failed for chat task %s (attempt %s)", task_id, attempt)
            audit("explanation_agent", "explain", "error", attempt=attempt, error_type=type(exc).__name__)
            return fallback(EXPLANATION_UNAVAILABLE_MESSAGE)

        draft = build_draft_response(results, explained, tests_with_sources)
        try:
            decision = safety_service(
                SafetyValidateRequest(
                    task_id=task_id,
                    report_id=report_id,
                    user_id=user_id,
                    original_result=[LabResult.model_validate(r.model_dump()) for r in results],
                    retrieved_sources=relevant_sources,
                    draft_response=draft,
                )
            )
        except Exception as exc:
            logger.exception("Safety validation failed for chat task %s (attempt %s)", task_id, attempt)
            audit("safety_agent", "validate", "error", attempt=attempt, error_type=type(exc).__name__)
            return fallback(FALLBACK_MESSAGE)

        if isinstance(decision, ApprovedResponse) and decision.approved:
            audit("safety_agent", "validate", "approved", attempt=attempt)
            audit("coordinator", "answer_question", "approved", outcome="approved")
            return ChatAnswer(
                task_id=task_id,
                status="approved",
                findings=_approved_findings(results, explained, tests_with_sources, by_test),
            )

        reason = getattr(decision, "reason", "unknown")
        audit("safety_agent", "validate", "rejected", attempt=attempt, reason=reason)
        rejection_feedback.append(rejection_note(reason))

    return fallback(FALLBACK_MESSAGE)


def _approved_findings(results, explained, tests_with_sources, by_test) -> list[AnswerFinding]:
    """Split the approved draft back into per-test fields, mirroring build_draft_response.

    Text the draft left out (agent prose for a test without sources) is left out here too.
    """
    findings = []
    for index, result in enumerate(results):
        item = _match_explained(index, result, explained)
        has_sources = result.test.lower() in tests_with_sources
        finding = AnswerFinding(
            test=result.test,
            value=result.value,
            unit=result.unit,
            reference_range=result.reference_range,
            status=result.status,
            insufficient_information=not has_sources or bool(item and item.insufficient_information),
            generation_mode=item.generation_mode if item else None,
            sources=source_links(by_test.get(result.test.lower())),
        )
        if item and (has_sources or item.insufficient_information):
            finding.what_it_measures = item.what_it_measures
            finding.explanation = item.explanation
            finding.possible_meaning = item.possible_meaning
            finding.recommended_discussion = item.recommended_discussion
        if not has_sources:
            finding.explanation = " ".join(filter(None, [INSUFFICIENT_INFORMATION_MESSAGE, finding.explanation]))
        findings.append(finding)
    return findings
