"""Generate a grounded explanation per finding, or an explicit safe response when it cannot."""

import os
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from explanation_agent.copy import (
    conflict_sections,
    display_result,
    insufficient_sections,
    safe_fallback_sections,
    template_sections,
    unavailable_sections,
)
from explanation_agent.grounding import passages_for
from explanation_agent.guardrails import (
    DraftParseError,
    disallowed_reasons,
    parse_model_draft,
    personal_disallowed_reasons,
    validate_model_draft,
)
from explanation_agent.llm import (
    ExplanationModelError,
    TemplateExplanationClient,
    build_gemini_client_from_env,
)
from explanation_agent.models import (
    ExplainedFinding,
    ExplanationRequest,
    ExplanationResponse,
    ExplanationTask,
    GenerationMode,
    Passage,
)
from explanation_agent.prompt import SYSTEM_PROMPT, build_user_prompt

ServiceMode = Literal["llm", "template", "unavailable"]
_TEMPLATE_EXCERPT_LIMIT = 500


class CompletionClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the raw model text."""


@dataclass
class ExplanationService:
    mode: ServiceMode
    client: CompletionClient | None = None

    def explain(self, request: ExplanationRequest) -> ExplanationResponse:
        findings: list[ExplainedFinding] = []
        notes: list[str] = []
        conflicting = _conflicting_tests(request)
        for finding in request.findings:
            task = ExplanationTask(
                test_name=finding.test,
                value=finding.value,
                unit=finding.unit,
                reference_range=finding.reference_range,
                status=finding.status,
                passages=passages_for(finding.test, request.retrieved_sources),
                rejection_feedback=request.rejection_feedback,
                user_question=request.user_question,
            )
            if finding.test.casefold() in conflicting:
                explained = _explained(task, conflict_sections(task), [], "insufficient")
                note = "conflicting values were reported for this test; none was interpreted"
            else:
                explained, note = self._explain_one(task)
            findings.append(explained)
            if note:
                notes.append(f"{finding.test}: {note}")
        return ExplanationResponse(
            task_id=request.task_id,
            report_id=request.report_id,
            user_id=request.user_id,
            findings=findings,
            safety_notes=notes,
        )

    def _explain_one(self, task: ExplanationTask) -> tuple[ExplainedFinding, str | None]:
        if not task.passages:
            return (
                _explained(task, insufficient_sections(task), [], "insufficient"),
                "no retrieved sources with a citation",
            )
        if self.mode == "unavailable" or (self.client is None and self.mode != "template"):
            return (
                _explained(task, unavailable_sections(task), [], "unavailable"),
                "explanation model unavailable",
            )
        if self.mode == "template":
            return self._from_template(task)
        return self._from_model(task)

    def _from_template(self, task: ExplanationTask) -> tuple[ExplainedFinding, str | None]:
        quotes = _safe_quotes(task.passages)
        if not quotes:
            return (
                _explained(task, insufficient_sections(task), [], "insufficient"),
                "every retrieved source contained disallowed language",
            )
        sections = template_sections(task, quotes)
        titles = [quote.title for quote in quotes]
        reasons = _section_reasons(task, sections, quotes)
        if reasons:
            return _fallback(task, reasons)
        return _explained(task, sections, titles, "template"), None

    def _from_model(self, task: ExplanationTask) -> tuple[ExplainedFinding, str | None]:
        assert self.client is not None
        extra_feedback: list[str] = []
        for attempt in range(2):
            user_prompt = build_user_prompt(task, extra_feedback if attempt else None)
            try:
                raw = self.client.complete(SYSTEM_PROMPT, user_prompt)
                draft = parse_model_draft(raw)
            except ExplanationModelError:
                return (
                    _explained(task, unavailable_sections(task), [], "unavailable"),
                    "explanation model request failed",
                )
            except DraftParseError as exc:
                extra_feedback = [str(exc)]
                continue

            if draft.insufficient_information:
                return (
                    _explained(task, insufficient_sections(task), [], "insufficient"),
                    "model reported insufficient information in the sources",
                )

            reasons = validate_model_draft(draft, task)
            if not reasons:
                sections = {
                    "what_it_measures": draft.what_it_measures.strip(),
                    "explanation": draft.explanation.strip(),
                    "possible_meaning": draft.possible_meaning.strip(),
                    "recommended_discussion": draft.recommended_discussion.strip(),
                }
                return _explained(task, sections, draft.sources_used, "llm"), None
            extra_feedback = reasons

        return _fallback(task, extra_feedback)


def _explained(
    task: ExplanationTask,
    sections: dict[str, str],
    sources_used: list[str],
    generation_mode: GenerationMode,
) -> ExplainedFinding:
    return ExplainedFinding(
        test_name=task.test_name,
        result=display_result(task.value, task.unit),
        status=task.status,
        what_it_measures=sections["what_it_measures"],
        explanation=sections["explanation"],
        possible_meaning=sections["possible_meaning"],
        recommended_discussion=sections["recommended_discussion"],
        insufficient_information=generation_mode not in ("llm", "template"),
        sources_used=sources_used,
        generation_mode=generation_mode,
    )


def _fallback(task: ExplanationTask, reasons: list[str]) -> tuple[ExplainedFinding, str]:
    detail = "; ".join(reasons) if reasons else "draft could not be verified"
    return (
        _explained(task, safe_fallback_sections(task), [], "safe_fallback"),
        f"draft replaced with a safe response ({detail})",
    )


def _conflicting_tests(request: ExplanationRequest) -> set[str]:
    """Test names (casefolded) that appear with more than one value or unit."""
    seen: dict[str, set[tuple[float, str | None]]] = {}
    for finding in request.findings:
        seen.setdefault(finding.test.casefold(), set()).add((finding.value, finding.unit))
    return {name for name, readings in seen.items() if len(readings) > 1}


def build_explanation_service() -> ExplanationService:
    provider = os.getenv("EXPLANATION_PROVIDER", "gemini").strip().lower()
    if provider == "template":
        return ExplanationService(mode="template", client=TemplateExplanationClient())
    client = build_gemini_client_from_env()
    if client is None:
        return ExplanationService(mode="unavailable", client=None)
    return ExplanationService(mode="llm", client=client)


def _safe_quotes(passages: list[Passage]) -> list[Passage]:
    quotes: list[Passage] = []
    for passage in passages:
        excerpt = " ".join(passage.excerpt.split())
        if len(excerpt) > _TEMPLATE_EXCERPT_LIMIT:
            excerpt = excerpt[:_TEMPLATE_EXCERPT_LIMIT].rstrip() + "..."
        # Reference text may use clinical words in the third person; only statements
        # about the reader disqualify an excerpt.
        if personal_disallowed_reasons(f"{passage.title} {excerpt}"):
            continue
        quotes.append(Passage(title=passage.title, url=passage.url, excerpt=excerpt))
    return quotes


def _section_reasons(task: ExplanationTask, sections: dict[str, str], quotes: list[Passage]) -> list[str]:
    body = "\n".join(sections.values())
    # The template's own sentences (titles included), rebuilt with every excerpt
    # blanked out, get the full check. The quoted excerpts are third-person reference
    # text, so they only get the personal check.
    authored = "\n".join(template_sections(task, [replace(q, excerpt="") for q in quotes]).values())
    reasons = disallowed_reasons(authored)
    for quote in quotes:
        reasons.extend(r for r in personal_disallowed_reasons(quote.excerpt) if r not in reasons)
    for quote in quotes:
        if quote.title not in body:
            reasons.append(f"the draft does not mention the retrieved source: {quote.title}")
    return reasons
