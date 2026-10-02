"""Patient-facing sentences shared by the prompt, fallbacks, and validators.

Agent-authored text avoids diagnostic and treatment claims. The disclaimer is not
added here: the Coordinator appends the Safety Agent's ``DISCLAIMER`` to the draft.
"""

from explanation_agent.models import ExplanationTask, Passage, Status

RECOMMENDED_DISCUSSION = (
    "Discuss this result with a qualified healthcare professional. "
    "No medication or treatment advice is provided."
)

_NOT_ENOUGH = "There is not enough reliable information"


def display_value(value: float) -> str:
    """11.2 -> "11.2", 250.0 -> "250", never scientific notation."""

    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text or "0"


def display_result(value: float, unit: str | None) -> str:
    number = display_value(value)
    return f"{number} {unit}" if unit else number


def recorded_result_sentence(value: float, unit: str | None) -> str:
    return f"The recorded result is {display_result(value, unit)}."


def status_sentence(status: Status | None) -> str:
    if status is None:
        return "No status could be determined from the supplied reference range."
    return f"The supplied status is {status}."


def range_sentence(reference_range: str | None) -> str:
    if reference_range and reference_range.strip():
        return f"The reference range supplied is {reference_range.strip()}."
    return "No reference range was supplied."


def required_explanation_sentences(task: ExplanationTask) -> tuple[str, str, str]:
    return (
        recorded_result_sentence(task.value, task.unit),
        range_sentence(task.reference_range),
        status_sentence(task.status),
    )


def factual_explanation(task: ExplanationTask, closing: str) -> str:
    recorded, supplied_range, supplied_status = required_explanation_sentences(task)
    return f"{recorded} {supplied_range} {supplied_status} {closing}"


def insufficient_sections(task: ExplanationTask) -> dict[str, str]:
    follow_up = ""
    if task.user_question:
        follow_up = (
            " A follow-up question was submitted. "
            "There is not enough reliable information to answer it."
        )
    return {
        "what_it_measures": (
            f"{_NOT_ENOUGH} in the retrieved sources to describe what "
            f"{task.test_name} measures."
        ),
        "explanation": factual_explanation(
            task,
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


def unavailable_sections(task: ExplanationTask) -> dict[str, str]:
    return {
        "what_it_measures": (
            f"The explanation model is unavailable, so what {task.test_name} "
            "measures was not generated from the retrieved sources."
        ),
        "explanation": factual_explanation(
            task,
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


def conflict_sections(task: ExplanationTask) -> dict[str, str]:
    return {
        "what_it_measures": (
            f"The report lists more than one different value for {task.test_name}, "
            "so the values conflict."
        ),
        "explanation": factual_explanation(
            task,
            (
                "Because the report lists conflicting values for this test, no single value "
                "was interpreted. The recorded result was not changed."
            ),
        ),
        "possible_meaning": (
            f"{_NOT_ENOUGH} to describe a possible meaning while the values conflict. "
            "Please check the original report for the correct value. No personal condition is assigned."
        ),
        "recommended_discussion": RECOMMENDED_DISCUSSION,
    }


def safe_fallback_sections(task: ExplanationTask) -> dict[str, str]:
    return {
        "what_it_measures": (
            f"A grounded explanation of what {task.test_name} measures could "
            "not be verified against the retrieved sources."
        ),
        "explanation": factual_explanation(
            task,
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


def template_sections(task: ExplanationTask, quotes: list[Passage]) -> dict[str, str]:
    """Stitch only the supplied source text. Used when no model call is wanted."""

    titles = ", ".join(quote.title for quote in quotes)
    quoted = " ".join(f'From "{quote.title}": {quote.excerpt.strip()}' for quote in quotes)
    return {
        "what_it_measures": (
            f"What {task.test_name} measures is limited to the retrieved "
            f"sources named here: {titles}."
        ),
        "explanation": factual_explanation(
            task,
            "Those retrieved sources were used as the sole basis for this explanation.",
        ),
        "possible_meaning": (
            f"General educational context from the retrieved sources. {quoted} "
            "No personal condition is assigned."
        ),
        "recommended_discussion": RECOMMENDED_DISCUSSION,
    }
