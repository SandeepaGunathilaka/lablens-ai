"""Join Explanation Agent fields into the single draft the Safety Agent reads."""

from explanation_agent.copy import STANDARD_DISCLAIMER
from explanation_agent.models import ExplanationResponse


def build_draft_response(response: ExplanationResponse) -> str:
    """Join the four narrative fields and append the standard disclaimer once.

    The Coordinator should send this string to the Safety Agent. Do not append
    the disclaimer again.
    """

    sections = (
        ("What it measures", response.what_it_measures),
        ("Explanation", response.explanation),
        ("Possible meaning", response.possible_meaning),
        ("Recommended discussion", response.recommended_discussion),
    )
    body = "\n\n".join(f"{heading}\n{text.strip()}" for heading, text in sections)
    return f"{body}\n\n{STANDARD_DISCLAIMER}"
