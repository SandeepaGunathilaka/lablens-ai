"""Generate a grounded explanation, or an explicit safe response when it cannot."""

import logging
import os
from dataclasses import dataclass
from typing import Literal, Protocol

from explanation_agent.copy import (
    insufficient_sections,
    safe_fallback_sections,
    template_sections,
    unavailable_sections,
)
from explanation_agent.guardrails import (
    DraftParseError,
    disallowed_reasons,
    parse_model_draft,
    validate_model_draft,
)
from explanation_agent.llm import (
    ExplanationModelError,
    TemplateExplanationClient,
    build_openai_client_from_env,
)
from explanation_agent.models import (
    ExplanationRequest,
    ExplanationResponse,
    GenerationMode,
    RetrievedSource,
    Status,
)
from explanation_agent.prompt import SYSTEM_PROMPT, build_user_prompt
from explanation_agent.status import StatusAssessment, calculate_status

logger = logging.getLogger("lablens.explanation")

ServiceMode = Literal["llm", "template", "unavailable"]
_EXCERPT_LIMIT = 500


class CompletionClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the raw model text."""


@dataclass
class ExplanationService:
    mode: ServiceMode
    client: CompletionClient | None = None

    def explain(self, request: ExplanationRequest) -> ExplanationResponse:
        assessment = resolve_status(request)
        regenerated = bool(request.rejection_feedback)

        if not request.retrieved_sources:
            response = self._respond(
                request,
                assessment,
                insufficient_sections(request, assessment.status),
                insufficient=True,
                sources_used=[],
                generation_mode="insufficient",
                regenerated=regenerated,
            )
            self._audit(response)
            return response

        if self.mode == "unavailable" or (self.client is None and self.mode != "template"):
            response = self._respond(
                request,
                assessment,
                unavailable_sections(request, assessment.status),
                insufficient=True,
                sources_used=[],
                generation_mode="unavailable",
                regenerated=regenerated,
            )
            self._audit(response)
            return response

        if self.mode == "template":
            response = self._from_template(request, assessment, regenerated)
            self._audit(response)
            return response

        response = self._from_model(request, assessment, regenerated)
        self._audit(response)
        return response

    def _from_template(
        self,
        request: ExplanationRequest,
        assessment: StatusAssessment,
        regenerated: bool,
    ) -> ExplanationResponse:
        quotes = _safe_quotes(request.retrieved_sources)
        if not quotes:
            return self._respond(
                request,
                assessment,
                insufficient_sections(request, assessment.status),
                insufficient=True,
                sources_used=[],
                generation_mode="insufficient",
                regenerated=regenerated,
            )
        sections = template_sections(request, assessment.status, quotes)
        titles = [title for title, _excerpt in quotes]
        reasons = _section_reasons(sections, titles)
        if reasons:
            return self._fallback(request, assessment, regenerated or True)
        return self._respond(
            request,
            assessment,
            sections,
            insufficient=False,
            sources_used=titles,
            generation_mode="template",
            regenerated=regenerated,
        )

    def _from_model(
        self,
        request: ExplanationRequest,
        assessment: StatusAssessment,
        regenerated: bool,
    ) -> ExplanationResponse:
        assert self.client is not None
        extra_feedback: list[str] = []
        for attempt in range(2):
            user_prompt = build_user_prompt(
                request,
                assessment.status,
                extra_feedback if attempt else None,
            )
            try:
                raw = self.client.complete(SYSTEM_PROMPT, user_prompt)
                draft = parse_model_draft(raw)
            except (ExplanationModelError, DraftParseError) as exc:
                if isinstance(exc, ExplanationModelError):
                    return self._respond(
                        request,
                        assessment,
                        unavailable_sections(request, assessment.status),
                        insufficient=True,
                        sources_used=[],
                        generation_mode="unavailable",
                        regenerated=regenerated or attempt > 0,
                    )
                extra_feedback = [str(exc)]
                continue

            if draft.insufficient_information:
                return self._respond(
                    request,
                    assessment,
                    insufficient_sections(request, assessment.status),
                    insufficient=True,
                    sources_used=[],
                    generation_mode="insufficient",
                    regenerated=regenerated or attempt > 0,
                )

            reasons = validate_model_draft(draft, request, assessment.status)
            if not reasons:
                return self._respond(
                    request,
                    assessment,
                    {
                        "what_it_measures": draft.what_it_measures.strip(),
                        "explanation": draft.explanation.strip(),
                        "possible_meaning": draft.possible_meaning.strip(),
                        "recommended_discussion": draft.recommended_discussion.strip(),
                    },
                    insufficient=False,
                    sources_used=draft.sources_used,
                    generation_mode="llm",
                    regenerated=regenerated or attempt > 0,
                )
            extra_feedback = reasons

        return self._fallback(request, assessment, True)

    def _fallback(
        self,
        request: ExplanationRequest,
        assessment: StatusAssessment,
        regenerated: bool,
    ) -> ExplanationResponse:
        return self._respond(
            request,
            assessment,
            safe_fallback_sections(request, assessment.status),
            insufficient=True,
            sources_used=[],
            generation_mode="safe_fallback",
            regenerated=regenerated,
        )

    def _respond(
        self,
        request: ExplanationRequest,
        assessment: StatusAssessment,
        sections: dict[str, str],
        *,
        insufficient: bool,
        sources_used: list[str],
        generation_mode: GenerationMode,
        regenerated: bool,
    ) -> ExplanationResponse:
        return ExplanationResponse(
            task_id=request.task_id,
            test_name=request.test_name,
            value=request.value,
            unit=request.unit,
            reference_range=request.reference_range,
            status=assessment.status,
            status_detail=assessment.detail,
            what_it_measures=sections["what_it_measures"],
            explanation=sections["explanation"],
            possible_meaning=sections["possible_meaning"],
            recommended_discussion=sections["recommended_discussion"],
            insufficient_information=insufficient,
            sources_used=sources_used,
            generation_mode=generation_mode,
            regenerated=regenerated,
        )

    def _audit(self, response: ExplanationResponse) -> None:
        logger.info(
            "audit agent=explanation action=generate task_id=%s result=%s status=%s insufficient=%s",
            response.task_id,
            response.generation_mode,
            response.status,
            response.insufficient_information,
        )


def resolve_status(request: ExplanationRequest) -> StatusAssessment:
    """Use the Coordinator's status when it is present. Otherwise calculate it."""

    if request.status is not None:
        return StatusAssessment(
            status=request.status,
            reason="status_provided",
            detail=f"Status is {request.status} because the caller supplied it.",
        )
    return calculate_status(request.value, request.reference_range)


def build_explanation_service() -> ExplanationService:
    provider = os.getenv("EXPLANATION_PROVIDER", "openai").strip().lower()
    if provider == "template":
        return ExplanationService(mode="template", client=TemplateExplanationClient())
    client = build_openai_client_from_env()
    if client is None:
        return ExplanationService(mode="unavailable", client=None)
    return ExplanationService(mode="llm", client=client)


def _safe_quotes(sources: list[RetrievedSource]) -> list[tuple[str, str]]:
    quotes: list[tuple[str, str]] = []
    for source in sources:
        excerpt = " ".join(source.excerpt.split())
        if len(excerpt) > _EXCERPT_LIMIT:
            excerpt = excerpt[:_EXCERPT_LIMIT].rstrip() + "..."
        candidate = f"{source.title} {excerpt}"
        if disallowed_reasons(candidate):
            continue
        quotes.append((source.title, excerpt))
    return quotes


def _section_reasons(sections: dict[str, str], titles: list[str]) -> list[str]:
    body = "\n".join(sections.values())
    reasons = disallowed_reasons(body)
    for title in titles:
        if title not in body:
            reasons.append(f"the draft does not mention the retrieved source: {title}")
    return reasons
