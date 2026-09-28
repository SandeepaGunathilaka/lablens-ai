import pytest

from explanation_agent.status import calculate_status


@pytest.mark.parametrize(
    ("value", "reference_range", "status"),
    [
        ("14", "12-15", "normal"),
        ("12", "12-15", "normal"),
        ("15", "12.0-15.5", "normal"),
        ("11.9", "12-15", "low"),
        ("15.6", "12.0-15.5", "high"),
        ("14", "12 to 15", "normal"),
        ("14", "12–15", "normal"),
        ("10000", "4,500-11,000", "normal"),
        ("3000", "4,500-11,000", "low"),
        ("199.9", "<200", "normal"),
        ("200", "<200", "high"),
        ("200", "<=200", "normal"),
        ("200.1", "<= 200 mg/dL", "high"),
        ("40", ">40", "low"),
        ("40.1", ">40", "normal"),
        ("40", ">=40", "normal"),
        ("39.9", ">=40 mg/dL", "low"),
        ("50", ">=40 and <60", "normal"),
        ("40", ">=40 and <60", "normal"),
        ("60", ">=40 and <60", "high"),
        ("39", ">=40 and <60", "low"),
        (10.2, "12-15", "low"),
        ("1,200", "1,000-1,400", "normal"),
    ],
)
def test_calculate_status_known_ranges(value, reference_range, status):
    assert calculate_status(value, reference_range).status == status


@pytest.mark.parametrize(
    ("value", "reference_range", "reason"),
    [
        ("10", None, "range_missing"),
        ("10", "   ", "range_missing"),
        ("10", "see note", "range_unparseable"),
        ("10", "up to 200", "range_unparseable"),
        ("10", "Male: 13-17 Female: 12-15", "range_unparseable"),
        ("10", "15-12", "range_unparseable"),
        ("<0.1", "0-5", "value_unparseable"),
        ("negative", "0-5", "value_unparseable"),
        (None, "0-5", "value_unparseable"),
    ],
)
def test_calculate_status_does_not_guess(value, reference_range, reason):
    assessment = calculate_status(value, reference_range)

    assert assessment.status == "unknown"
    assert assessment.reason == reason
    assert "not guessed" in assessment.detail
