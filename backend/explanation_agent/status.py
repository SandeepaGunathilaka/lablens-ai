"""Deterministic low / normal / high status from a value and reference range.

The language model must not choose this status. The Coordinator should call
``calculate_status`` and pass ``status`` on ``ExplanationRequest``. Missing or
unreadable ranges stay ``unknown`` so nothing is guessed.
"""

import re
from dataclasses import dataclass
from typing import Literal

from explanation_agent.models import Status

StatusReason = Literal[
    "below_range",
    "above_range",
    "within_range",
    "above_upper_limit",
    "within_upper_limit",
    "below_lower_limit",
    "within_lower_limit",
    "range_missing",
    "range_unparseable",
    "value_unparseable",
    "status_provided",
]

_BETWEEN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:-|\bto\b)\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_UPPER = re.compile(r"(<=|<)\s*(-?\d+(?:\.\d+)?)")
_LOWER = re.compile(r"(>=|>)\s*(-?\d+(?:\.\d+)?)")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class StatusAssessment:
    status: Status
    reason: StatusReason
    detail: str


def calculate_status(value: object, reference_range: str | None) -> StatusAssessment:
    """Return low, normal, high, or unknown without guessing."""

    number = _parse_value(value)
    if number is None:
        return StatusAssessment(
            status="unknown",
            reason="value_unparseable",
            detail=(
                "Status is unknown because the value could not be read as a number. "
                "The status was not guessed."
            ),
        )

    prepared = _prepare_range(reference_range)
    if prepared is None:
        return StatusAssessment(
            status="unknown",
            reason="range_missing",
            detail=(
                "Status is unknown because the reference range is missing. "
                "The status was not guessed."
            ),
        )

    bounds = _parse_bounds(prepared)
    if bounds is None:
        return StatusAssessment(
            status="unknown",
            reason="range_unparseable",
            detail=(
                "Status is unknown because the reference range could not be parsed. "
                "The status was not guessed."
            ),
        )

    return _assess(number, bounds)


def _parse_value(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            return None
        return number
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not _NUMBER.fullmatch(text):
            return None
        return float(text)
    return None


def _prepare_range(reference_range: str | None) -> str | None:
    if reference_range is None:
        return None
    text = reference_range.strip()
    if not text:
        return None
    text = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace(",", "")
        .replace("≤", "<=")
        .replace("≥", ">=")
    )
    return text


def _parse_bounds(text: str) -> tuple[float | None, bool, float | None, bool] | None:
    between = _BETWEEN.findall(text)
    upper = _UPPER.findall(text)
    lower = _LOWER.findall(text)

    if between and (upper or lower):
        return None
    if len(between) > 1 or len(upper) > 1 or len(lower) > 1:
        return None

    if len(between) == 1:
        low = float(between[0][0])
        high = float(between[0][1])
        if low > high:
            return None
        return low, True, high, True

    if not upper and not lower:
        return None

    low_value = float(lower[0][1]) if lower else None
    low_inclusive = bool(lower and lower[0][0] == ">=")
    high_value = float(upper[0][1]) if upper else None
    high_inclusive = bool(upper and upper[0][0] == "<=")
    if low_value is not None and high_value is not None and low_value > high_value:
        return None
    return low_value, low_inclusive, high_value, high_inclusive


def _assess(
    number: float,
    bounds: tuple[float | None, bool, float | None, bool],
) -> StatusAssessment:
    low, low_inclusive, high, high_inclusive = bounds
    below = low is not None and (number < low or (number == low and not low_inclusive))
    above = high is not None and (number > high or (number == high and not high_inclusive))

    if below:
        reason: StatusReason = "below_range" if high is not None else "below_lower_limit"
        return StatusAssessment(
            status="low",
            reason=reason,
            detail="Status is low because the value is below the reference range.",
        )
    if above:
        reason = "above_range" if low is not None else "above_upper_limit"
        return StatusAssessment(
            status="high",
            reason=reason,
            detail="Status is high because the value is above the reference range.",
        )

    if low is not None and high is not None:
        detail = "Status is normal because the value is inside the reference range."
        reason = "within_range"
    elif high is not None:
        detail = "Status is normal because the value is within the upper limit."
        reason = "within_upper_limit"
    else:
        detail = "Status is normal because the value is within the lower limit."
        reason = "within_lower_limit"
    return StatusAssessment(status="normal", reason=reason, detail=detail)
