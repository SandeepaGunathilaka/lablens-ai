"""Curated lay phrases for approved tests, matched exactly after removing question filler.

A phrase belongs here only when the test's own curated record uses it (the LDL record
says "'bad' cholesterol", the HDL record "'good' cholesterol"), and each phrase names
exactly one canonical test. Anything else still goes to the semantic fallback.
"""

from agents.keyword_retriever import normalize_test_name

LAY_TERMS = {
    "bad cholesterol": "LDL",
    "good cholesterol": "HDL",
}
_LEADING_FILLER = ("what is the", "what is", "what's", "what does", "meaning of")
_TRAILING_FILLER = ("meaning", "means", "mean")


def match_lay_term(query: str) -> str | None:
    """Return the canonical test name for a lay phrase such as "bad cholesterol meaning"."""
    text = normalize_test_name(query).rstrip("?").strip()
    for prefix in _LEADING_FILLER:
        if text.startswith(prefix + " "):
            text = text[len(prefix) + 1:]
            break
    for suffix in _TRAILING_FILLER:
        if text.endswith(" " + suffix):
            text = text[: -len(suffix) - 1]
            break
    return LAY_TERMS.get(text)
