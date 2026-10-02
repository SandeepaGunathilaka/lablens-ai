"""Work out what a chat question is asking before any explanation is generated.

Deterministic rules only, so routing behaves the same with or without a language model:
diagnosis/treatment requests are referred to a doctor, small talk and off-topic messages get
a short reply, and questions about particular tests are narrowed to those tests.
"""

import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Literal

from agents.document_agent import TEST_ALIASES

Intent = Literal["diagnosis", "small_talk", "off_topic", "not_in_report", "tests", "report"]

REDIRECT_MESSAGE = (
    "I can't diagnose conditions or recommend medicines or treatment. LabLens can explain what your results "
    "measure, but only a qualified healthcare professional can tell you what they mean for your health. Please "
    "discuss this report with your doctor."
)

_LAY_TERMS = {
    "LDL Cholesterol": ("bad cholesterol",),
    "HDL Cholesterol": ("good cholesterol",),
    "Hemoglobin": ("haemoglobin",),
    "WBC": ("white cell", "white cells"),
    "RBC": ("red cell", "red cells"),
}

_ALIASES = sorted(
    ((alias, test) for test, names in TEST_ALIASES.items() for alias in (*names, test.lower(), *_LAY_TERMS.get(test, ()))),
    key=lambda pair: -len(pair[0]),
)

# Words a misspelling may be corrected to: test names plus the medical and report words the rules look for.
# Short words are excluded because a one-letter slip there usually makes a different real word.
_MIN_CORRECTABLE = 5
_SPELLING_CUTOFF = 0.85
_LONG_WORD, _LONG_WORD_CUTOFF = 8, 0.8
_VOCABULARY = sorted({
    word for alias, _ in _ALIASES for word in alias.split() if len(word) >= _MIN_CORRECTABLE
} | {
    "diagnose", "diagnosis", "disease", "illness", "condition", "disorder", "cancer", "leukemia", "leukaemia",
    "anemia", "anaemia", "anemic", "anaemic", "diabetes", "diabetic", "infection", "medicine", "medication",
    "medications", "drugs", "pills", "dosage", "supplement", "supplements", "prescribe", "prescription", "treatment",
    "report", "result", "results", "value", "values", "level", "levels", "range", "explain", "summary", "normal",
    "abnormal", "blood", "meaning", "finding", "findings", "worry", "concern",
})

_DIAGNOSIS = re.compile(
    r"\bdo i have\b|\bdiagnos|\bam i (?:sick|ill|unwell|healthy|ok|okay|fine|dying|anemic|anaemic|diabetic)\b"
    r"|\bwhether i(?:'m| am) (?:sick|ill|healthy)\b|\bwhat(?:'s| is) wrong with me\b"
    r"|\bwhat (?:disease|illness|condition|disorder)\b|\b(?:is|could) (?:it|this|that) be\b"
    r"|\b(?:cancer|leukemia|leukaemia|anemia|anaemia|diabetes|infection|disease)\b"
    r"|\b(?:medicine|medication|medications|drug|drugs|pill|pills|dose|dosage|supplement|supplements|prescri\w*)\b"
    r"|\b(?:treat|treatment|cure)\b|\bshould i take\b"
    r"|\bhow (?:do|can|should) i (?:fix|treat|cure|lower|raise|increase|reduce|improve|bring)\b"
)

_SMALL_TALK = {
    "greeting": re.compile(r"^(?:hi|hello|hey|hiya|good (?:morning|afternoon|evening))\b[\s!.]*$"),
    "thanks": re.compile(r"^(?:thanks|thank you|thx|ty|cheers)\b.*$"),
    "ack": re.compile(r"^(?:k|kk|ok|okay|okey|alright|got it|cool|nice|great|fine|sure|yes|no|yep|nope|hmm+|ah|oh)[\s!.]*$"),
}

_ABOUT_REPORT = re.compile(
    r"\b(?:report|result|results|value|values|test|tests|level|levels|number|numbers|range|ranges|mean|means|meaning"
    r"|explain|summar\w*|normal|high|low|abnormal|blood|lab|this|these|that|it|my|answer|source|sources|based"
    r"|limitation\w*|why|finding\w*|worry|concern\w*)\b"
)


@dataclass(frozen=True)
class ChatIntent:
    kind: Intent
    tests: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    reply: str | None = None


def _correct(word: str) -> str:
    if len(word) < _MIN_CORRECTABLE or word in _VOCABULARY:
        return word
    cutoff = _LONG_WORD_CUTOFF if len(word) >= _LONG_WORD else _SPELLING_CUTOFF
    match = get_close_matches(word, _VOCABULARY, n=1, cutoff=cutoff)
    return match[0] if match else word


def _normalize(question: str) -> str:
    text = re.sub(r"\s+", " ", question.casefold().replace("’", "'")).strip()
    return re.sub(r"[a-z]+", lambda m: _correct(m.group()), text)


def mentioned_tests(question: str) -> list[str]:
    """Canonical test names the question refers to, longest alias first so 'ldl cholesterol' wins over 'cholesterol'."""
    text = _normalize(question)
    found: list[str] = []
    for alias, test in _ALIASES:
        pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
        if re.search(pattern, text):
            text = re.sub(pattern, " ", text)
            if test not in found:
                found.append(test)
    return found


def _example(report_tests: list[str]) -> str:
    return f"For example: 'What does my {report_tests[0]} result mean?'" if report_tests else ""


def classify(question: str, report_tests: list[str], *, tests_selected: bool = False) -> ChatIntent:
    """`tests_selected` means the user picked tests explicitly, so the question is about them even if it names none."""
    text = _normalize(question)
    report_tests = list(dict.fromkeys(report_tests))
    if _DIAGNOSIS.search(text):
        return ChatIntent("diagnosis", reply=f"{REDIRECT_MESSAGE} {_example(report_tests)}".strip())

    for kind, pattern in _SMALL_TALK.items():
        if pattern.match(text):
            opener = {"greeting": "Hello!", "thanks": "You're welcome!", "ack": "Okay."}[kind]
            return ChatIntent("small_talk", reply=f"{opener} Ask me about any result in this report. {_example(report_tests)}".strip())

    in_report = {t.lower(): t for t in report_tests}
    named = mentioned_tests(question)
    present = [in_report[t.lower()] for t in named if t.lower() in in_report]
    missing = [t for t in named if t.lower() not in in_report]
    if present:
        return ChatIntent("tests", tests=present, missing=missing)
    if missing:
        return ChatIntent("not_in_report", missing=missing, reply=(
            f"This report doesn't include {', '.join(missing)}. I can explain the tests it does include: "
            f"{', '.join(report_tests)}."))

    # The keyword rules are English-only, so other scripts are answered rather than declined.
    if tests_selected or _ABOUT_REPORT.search(text) or re.search(r"[^\x00-\x7f]", text):
        return ChatIntent("report", tests=list(report_tests))
    return ChatIntent("off_topic", reply=(
        f"I can only answer questions about the lab results in this report ({', '.join(report_tests)}). "
        f"{_example(report_tests)}").strip())
