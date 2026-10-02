"""Evidence capture for the Explanation Agent validation suite (Report 3, VAL-01..VAL-15).

Every test records its input, each expected-vs-actual check and an excerpt of the real
output. At the end of the session the records, the pytest outcome of each test and the
environment details are written to ``evidence/results.json`` for ``render_evidence.py``.
"""

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

EVIDENCE_DIR = Path(__file__).parent / "evidence"
BACKEND_DIR = Path(__file__).resolve().parents[2]

_RECORDS: list["Evidence"] = []


def _show(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


@dataclass
class Check:
    description: str
    expected: str
    actual: str
    passed: bool


@dataclass
class Evidence:
    val_id: str
    title: str
    control: str
    acceptance: str
    nodeid: str = ""
    inputs: dict = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    outputs: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    finding: str | None = None
    outcome: str = "not run"

    def given(self, **inputs) -> None:
        self.inputs.update({k: _show(v) for k, v in inputs.items()})

    def output(self, label: str, value) -> None:
        self.outputs[label] = _show(value)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def check(self, description: str, expected, actual, passed: bool | None = None) -> bool:
        ok = (expected == actual) if passed is None else bool(passed)
        self.checks.append(Check(description, _show(expected), _show(actual), ok))
        return ok

    def contains(self, description: str, text: str, needle: str) -> bool:
        found = needle in text
        return self.check(description, f'contains "{needle}"', f'contains "{needle}"' if found else f'"{needle}" not found', found)

    def absent(self, description: str, text: str, needle: str) -> bool:
        found = needle.lower() in text.lower()
        return self.check(description, f'no "{needle}"', f'found "{needle}"' if found else f'no "{needle}"', not found)

    def verify(self) -> None:
        failed = [c for c in self.checks if not c.passed]
        assert not failed, "; ".join(f"{c.description}: expected {c.expected}, got {c.actual}" for c in failed)


@pytest.fixture
def evidence(request):
    """evidence("VAL-01", title, control, acceptance) -> an Evidence record for this test."""

    def make(val_id: str, title: str, control: str, acceptance: str, finding: str | None = None) -> Evidence:
        record = Evidence(val_id, title, control, acceptance, nodeid=request.node.nodeid, finding=finding)
        request.node.stash.setdefault(_STASH_KEY, []).append(record)
        _RECORDS.append(record)
        return record

    return make


_STASH_KEY = pytest.StashKey[list]()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    records = item.stash.get(_STASH_KEY, [])
    if not records or (report.when != "call" and report.passed):
        return
    if hasattr(report, "wasxfail"):
        outcome = "xfailed" if report.skipped else "xpassed"
    else:
        outcome = report.outcome
    for record in records:
        record.outcome = outcome


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=BACKEND_DIR, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _environment() -> dict:
    from explanation_agent.llm import DEFAULT_MODEL
    from explanation_agent.prompt import SYSTEM_PROMPT

    kb = hashlib.sha256()
    for path in sorted((BACKEND_DIR / "data" / "knowledge_base").glob("*.json")):
        kb.update(path.read_bytes())
    return {
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": _git("rev-parse", "HEAD"),
        "llm_provider": f"Scripted stand-in for Gemini (configured default model: {DEFAULT_MODEL}); no network calls",
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16],
        "knowledge_base_sha256": kb.hexdigest()[:16],
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "pytest": pytest.__version__,
        "executed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }


def pytest_sessionfinish(session, exitstatus):
    if not _RECORDS:
        return
    EVIDENCE_DIR.mkdir(exist_ok=True)
    payload = {"environment": _environment(), "records": [asdict(r) for r in _RECORDS]}
    (EVIDENCE_DIR / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
