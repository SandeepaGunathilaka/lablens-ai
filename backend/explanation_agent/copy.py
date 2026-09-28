"""Patient-facing sentences shared by the prompt, fallbacks, and validators.

Agent-authored text avoids diagnostic and treatment claims. The Coordinator
appends ``STANDARD_DISCLAIMER`` after joining the four narrative fields.
"""

from explanation_agent.models import ExplanationRequest, Status

STANDARD_DISCLAIMER = (
    "This information is for general education only. It is not medical advice "
    "and it does not recommend treatment, medicines, or any change to care. "
    "Discuss your results with a qualified healthcare professional."
)

RECOMMENDED_DISCUSSION = (
    "Discuss this result with a qualified healthcare professional. "
    "No medication or treatment advice is provided."
)

_NOT_ENOUGH = "There is not enough reliable information"


def display_value(value: float | int | str) -> str:
    """Keep a string value exactly as reported. Format bare numbers stably."""

    if isinstance(value, str):
        return value.strip()
    return format(value, "g")


def recorded_result_sentence(value: float | int | str, unit: str) -> str:
    return f"The recorded result is {display_value(value)} {unit}."


def status_sentence(status: Status) -> str:
    return f"The supplied status is {status}."


def range_sentence(reference_range: str | None) -> str:
    if reference_range and reference_range.strip():
        return f"The reference range supplied is {reference_range.strip()}."
    return "No reference range was supplied."


def required_explanation_sentences(
    value: float | int | str,
    unit: str,
    reference_range: str | None,
    status: Status,
) -> tuple[str, str, str]:
    return (
        recorded_result_sentence(value, unit),
        range_sentence(reference_range),
        status_sentence(status),
    )


def factual_explanation(
    value: float | int | str,
    unit: str,
    reference_range: str | None,
    status: Status,
    closing: str,
) -> str:
    recorded, supplied_range, supplied_status = required_explanation_sentences(
        value, unit, reference_range, status
    )
    return f"{recorded} {supplied_range} {supplied_status} {closing}"


def insufficient_sections(request: ExplanationRequest, status: Status) -> dict[str, str]:
    follow_up = ""
    if request.user_question:
        follow_up = (
            " A follow-up question was submitted. "
            "There is not enough reliable information to answer it."
        )
    return {
        "what_it_measures": (
            f"{_NOT_ENOUGH} in the retrieved sources to describe what "
            f"{request.test_name} measures."
        ),
        "explanation": factual_explanation(
            request.value,
            request.unit,
            request.reference_range,
            status,
            (
                f"{_NOT_ENOUGH} to explain this result further. "
                "The recorded result was not changed."
            ),
        ),
        "possible_meaning": (
            f"{_NOT_ENOUGH} to describe a possible meaning. "
            f"No personal condition is assigned.{follow_up}"
        ),
        "recommended_discussion": RECOMMENDED_DISCUSSION,
    }


def unavailable_sections(request: ExplanationRequest, status: Status) -> dict[str, str]:
    return {
        "what_it_measures": (
            f"The explanation model is unavailable, so what {request.test_name} "
            "measures was not generated from the retrieved sources."
        ),
        "explanation": factual_explanation(
            request.value,
            request.unit,
            request.reference_range,
            status,
            (
                "The explanation model is unavailable, so no further explanation "
                "was generated and no medical claim was added."
            ),
        ),
        "possible_meaning": (
            f"{_NOT_ENOUGH} to describe a possible meaning because the explanation "
            "model is unavailable. No personal condition is assigned."
        ),
        "recommended_discussion": RECOMMENDED_DISCUSSION,
    }


def safe_fallback_sections(request: ExplanationRequest, status: Status) -> dict[str, str]:
    return {
        "what_it_measures": (
            f"A grounded explanation of what {request.test_name} measures could "
            "not be verified against the retrieved sources."
        ),
        "explanation": factual_explanation(
            request.value,
            request.unit,
            request.reference_range,
            status,
            (
                "The generated explanation was not used because it could not be "
                "verified. No unsupported medical claim was added."
            ),
        ),
        "possible_meaning": (
            f"{_NOT_ENOUGH} to describe a possible meaning without unsupported "
            "medical claims. No personal condition is assigned."
        ),
        "recommended_discussion": RECOMMENDED_DISCUSSION,
    }


def template_sections(
    request: ExplanationRequest,
    status: Status,
    quotes: list[tuple[str, str]],
) -> dict[str, str]:
    """Stitch only the supplied source text. Used when no model call is wanted."""

    titles = ", ".join(title for title, _excerpt in quotes)
    quoted = " ".join(
        f'From "{title}": {excerpt.strip()}' for title, excerpt in quotes
    )
    return {
        "what_it_measures": (
            f"What {request.test_name} measures is limited to the retrieved "
            f"sources named here: {titles}."
        ),
        "explanation": factual_explanation(
            request.value,
            request.unit,
            request.reference_range,
            status,
            "Those retrieved sources were used as the sole basis for this explanation.",
        ),
        "possible_meaning": (
            f"General educational context from the retrieved sources. {quoted} "
            "No personal condition is assigned."
        ),
        "recommended_discussion": RECOMMENDED_DISCUSSION,
    }
