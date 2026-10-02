"""Shared evidence capture and rendering for the assessment suites.

A suite's conftest imports ``evidence`` and ``pytest_runtest_makereport`` from here and
calls ``write_results`` from its own ``pytest_sessionfinish``. Each test records its input,
every expected-vs-actual check and excerpts of the real output; ``render`` turns the
resulting ``evidence/results.json`` into HTML pages and full-page PNG screenshots.

Outcomes: a passing test is PASS; a strict-xfail test is a confirmed finding (FAIL); a test
that calls ``Evidence.inconclusive`` is INCONCLUSIVE; a passing test that called
``Evidence.weakness`` behaved as the assessment predicted, but the behaviour is a weakness.
"""

import html
import json
import platform
import re
import subprocess
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
_RECORDS: list["Evidence"] = []
_STASH_KEY = pytest.StashKey[list]()


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
    path: str = ""
    inputs: dict = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    outputs: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    finding: str | None = None
    weakness: bool = False
    outcome: str = "not run"

    def given(self, **inputs) -> None:
        self.inputs.update({k: _show(v) for k, v in inputs.items()})

    def output(self, label: str, value) -> None:
        self.outputs[label] = _show(value)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def image(self, path: Path, caption: str) -> None:
        self.images.append({"path": str(path), "caption": caption})

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

    def weakness_confirmed(self, finding: str) -> None:
        """The system behaved as the assessment predicted, and that behaviour is a weakness."""
        self.finding = finding
        self.weakness = True

    def inconclusive(self, reason: str) -> None:
        self.note(f"INCONCLUSIVE: {reason}")
        pytest.skip(reason)

    def verify(self) -> None:
        failed = [c for c in self.checks if not c.passed]
        assert not failed, "; ".join(f"{c.description}: expected {c.expected}, got {c.actual}" for c in failed)


@pytest.fixture
def evidence(request):
    """evidence("IR-01", title, control, acceptance) -> an Evidence record for this test."""

    def make(val_id: str, title: str, control: str, acceptance: str, finding: str | None = None) -> Evidence:
        record = Evidence(val_id, title, control, acceptance, nodeid=request.node.nodeid,
                          path=str(request.node.path), finding=finding)
        request.node.stash.setdefault(_STASH_KEY, []).append(record)
        _RECORDS.append(record)
        return record

    return make


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


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=BACKEND_DIR, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def base_environment() -> dict:
    return {
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": git("rev-parse", "HEAD"),
        "working_tree": "uncommitted changes present" if git("status", "--porcelain") else "clean",
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "pytest": pytest.__version__,
        "executed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }


def write_results(suite_dir: Path, environment: dict) -> None:
    records = [r for r in _RECORDS if Path(r.path).resolve().is_relative_to(suite_dir.resolve())]
    if not records:
        return
    evidence_dir = suite_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    payload = {"environment": environment, "records": [asdict(r) for r in records]}
    (evidence_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# --- Rendering ------------------------------------------------------------------------------

CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 28px; background: #f4f6fa; color: #1d2433; width: 1180px; }
.card { background: #fff; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.08); padding: 24px 28px; }
h1 { margin: 0 0 4px; font-size: 24px; } h2 { font-size: 16px; margin: 22px 0 8px; color: #33415c; }
.meta { color: #5c677d; font-size: 13px; margin-bottom: 4px; }
.badge { display: inline-block; padding: 5px 14px; border-radius: 999px; font-weight: 700; font-size: 14px; color: #fff; }
.badge.pass { background: #1b9e5a; } .badge.fail { background: #d64545; } .badge.weak { background: #c77700; } .badge.inc { background: #6b7280; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { background: #24324d; color: #fff; text-align: left; padding: 8px 10px; }
td { border-bottom: 1px solid #e3e7ef; padding: 7px 10px; vertical-align: top; word-break: break-word; }
td:first-child { white-space: nowrap; }
td.res { white-space: nowrap; }
tr.ok td.res { color: #1b9e5a; font-weight: 700; } tr.bad td.res { color: #d64545; font-weight: 700; }
tr.weak td.res { color: #c77700; font-weight: 700; } tr.inc td.res { color: #6b7280; font-weight: 700; }
tr.bad { background: #fff1f1; } tr.weak { background: #fff8eb; }
td.exp { background: #eef5ff; } td.act { background: #f6fff4; } tr.bad td.act { background: #ffe3e3; }
pre { background: #0f1724; color: #d9e2f2; padding: 12px 14px; border-radius: 8px; font-size: 12px; white-space: pre-wrap; word-break: break-word; margin: 6px 0; }
.kv td:first-child { width: 230px; font-weight: 600; color: #33415c; }
.note { background: #fff8e1; border-left: 4px solid #f0b400; padding: 8px 12px; font-size: 13px; margin-top: 10px; }
.acc { background: #f0f3f9; border-left: 4px solid #24324d; padding: 8px 12px; font-size: 13px; }
figure { margin: 14px 0 0; } figure img { max-width: 100%; border: 1px solid #d5dbe6; border-radius: 8px; }
figcaption { font-size: 12px; color: #5c677d; margin-top: 4px; }
"""


def esc(value) -> str:
    return html.escape(str(value))


def page(title: str, body: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{esc(title)}</title><style>{CSS}</style></head><body>{body}</body></html>"


def status_of(record: dict) -> tuple[str, str, str]:
    """(short status, long label, css class) for one record."""
    outcome, finding = record["outcome"], record.get("finding")
    if outcome == "skipped":
        return "INCONCLUSIVE", "INCONCLUSIVE - not testable in this environment", "inc"
    if outcome == "xfailed":
        return "FAIL", f"FAIL - confirmed finding {finding}" if finding else "FAIL", "fail"
    if outcome == "failed":
        return "FAIL", "FAIL", "fail"
    if outcome in ("passed", "xpassed") and record.get("weakness"):
        return "PASS*", f"AS PREDICTED - weakness {finding} confirmed", "weak"
    if outcome in ("passed", "xpassed"):
        return "PASS", "PASS", "pass"
    return outcome.upper(), outcome.upper(), "fail"


def _record_page(record: dict, env: dict, cfg: dict) -> str:
    short, label, css = status_of(record)
    passed = sum(c["passed"] for c in record["checks"])
    rows = "".join(
        f"<tr class='{'ok' if c['passed'] else 'bad'}'><td>{i}</td><td>{esc(c['description'])}</td>"
        f"<td class='exp'>{esc(c['expected'])}</td><td class='act'>{esc(c['actual'])}</td>"
        f"<td class='res'>{'PASS' if c['passed'] else 'FAIL'}</td></tr>"
        for i, c in enumerate(record["checks"], start=1)
    )
    inputs = "".join(f"<tr><td>{esc(k)}</td><td><pre>{esc(v)}</pre></td></tr>" for k, v in record["inputs"].items())
    outputs = "".join(f"<h2>Actual output: {esc(k)}</h2><pre>{esc(v)}</pre>" for k, v in record["outputs"].items())
    notes = "".join(f"<div class='note'>{esc(n)}</div>" for n in record["notes"])
    images = "".join(
        f"<figure><img src='{Path(i['path']).resolve().as_uri()}'><figcaption>{esc(i['caption'])}</figcaption></figure>"
        for i in record.get("images", [])
    )
    footer = " | ".join(f"{esc(k)} {esc(env.get(v, ''))[:12] if k == 'Commit' else esc(env.get(v, ''))}" for k, v in cfg["footer"])
    body = f"""
<div class='card'>
  <div style='display:flex;justify-content:space-between;align-items:center'>
    <div><h1>{esc(record['val_id'])} - {esc(record['title'])}</h1>
    <div class='meta'>Control under test: {esc(record['control'])}</div>
    <div class='meta'>Test: {esc(record['nodeid'])}</div></div>
    <div style='text-align:right'><span class='badge {css}'>{esc(label)}</span>
    <div class='meta' style='margin-top:6px'>{passed}/{len(record['checks'])} checks met</div></div>
  </div>
  <h2>{esc(cfg['expected_heading'])}</h2><div class='acc'>{esc(record['acceptance'])}</div>
  <h2>Test input</h2><table class='kv'>{inputs}</table>
  <h2>Expected result vs actual result</h2>
  <table><tr><th>#</th><th>Check</th><th>Expected result</th><th>Actual result</th><th>Result</th></tr>{rows}</table>
  {outputs}{notes}{images}
  <div class='meta' style='margin-top:18px'>{footer}</div>
</div>"""
    return page(record["val_id"], body)


def _summary_page(records: list[dict], env: dict, cfg: dict) -> str:
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for record in records:
        groups.setdefault(re.match(cfg["id_regex"], record["val_id"]).group(0), []).append(record)

    def group_status(rs):
        shorts = {status_of(r)[0] for r in rs}
        for s in ("FAIL", "INCONCLUSIVE", "PASS*"):
            if s in shorts:
                return s
        return "PASS"

    statuses = {gid: group_status(rs) for gid, rs in groups.items()}
    count = lambda s: sum(v == s for v in statuses.values())  # noqa: E731
    executed = len(groups)
    findings = sorted({r["finding"] for r in records if r.get("finding") and status_of(r)[0] in ("FAIL", "PASS*")})
    rows = "".join(
        f"<tr class='{ {'pass': 'ok', 'fail': 'bad', 'weak': 'weak', 'inc': 'inc'}[status_of(r)[2]] }'>"
        f"<td>{esc(r['val_id'])}</td><td>{esc(r['title'])}</td><td>{esc(r['acceptance'])}</td>"
        f"<td>{sum(c['passed'] for c in r['checks'])}/{len(r['checks'])}</td>"
        f"<td class='res'>{esc(status_of(r)[0])}</td><td>{esc(r.get('finding') or '')}</td></tr>"
        for r in records
    )
    env_rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in env.items())
    legend = ("PASS* = the system behaved exactly as the assessment predicted, but that behaviour is a documented "
              "weakness carried into the risk register.") if count("PASS*") else ""
    body = f"""
<div class='card'>
  <h1>{esc(cfg['title'])}</h1>
  <div class='meta'>{esc(cfg['subtitle'])}</div>
  <h2>Results summary</h2>
  <table class='kv'>
    <tr><td>Total tests planned</td><td>{cfg['planned']}</td></tr>
    <tr><td>Tests executed</td><td>{executed} ({len(records)} test functions incl. sub-cases)</td></tr>
    <tr><td>Passed</td><td>{count('PASS')}</td></tr>
    <tr><td>Passed with documented weakness (PASS*)</td><td>{count('PASS*')}</td></tr>
    <tr><td>Failed (confirmed findings)</td><td>{count('FAIL')}</td></tr>
    <tr><td>Inconclusive</td><td>{count('INCONCLUSIVE')}</td></tr>
    <tr><td>Findings / weaknesses</td><td>{esc(', '.join(findings) or 'None')}</td></tr>
  </table>
  <h2>Test cases</h2>
  <table><tr><th>ID</th><th>Test</th><th>Expected result</th><th>Checks met</th><th>Status</th><th>Finding</th></tr>{rows}</table>
  <div class='meta' style='margin-top:8px'>{esc(legend)}</div>
  <h2>Test environment</h2><table class='kv'>{env_rows}</table>
</div>"""
    return page("Summary", body)


def _terminal_page(output: str) -> str:
    colored = esc(output)
    colored = re.sub(r"\b(PASSED)\b", r"<span style='color:#4ade80'>\1</span>", colored)
    colored = re.sub(r"\b(XFAIL|FAILED|ERROR)\b", r"<span style='color:#f87171'>\1</span>", colored)
    colored = re.sub(r"\b(SKIPPED)\b", r"<span style='color:#facc15'>\1</span>", colored)
    body = ("<div style='background:#0c0c0c;border-radius:8px;padding:16px'><div style='color:#9ca3af;font-size:12px;"
            "margin-bottom:8px'>PowerShell - backend</div><pre style='background:#0c0c0c;font-family:Consolas,monospace;"
            f"font-size:12.5px'>{colored}</pre></div>")
    return page("pytest output", body)


def render(suite_dir: Path, *, title: str, subtitle: str, planned: int, id_regex: str,
           expected_heading: str, footer: list[tuple[str, str]]) -> int:
    evidence_dir = suite_dir / "evidence"
    html_dir, shot_dir = evidence_dir / "html", evidence_dir / "screenshots"
    data = json.loads((evidence_dir / "results.json").read_text(encoding="utf-8"))
    env, records = data["environment"], data["records"]
    records.sort(key=lambda r: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", r["val_id"])])
    cfg = {"title": title, "subtitle": subtitle, "planned": planned, "id_regex": id_regex,
           "expected_heading": expected_heading, "footer": footer}
    html_dir.mkdir(parents=True, exist_ok=True)
    shot_dir.mkdir(parents=True, exist_ok=True)

    pages = {"00_summary": _summary_page(records, env, cfg)}
    output_file = evidence_dir / "pytest_output.txt"
    if output_file.exists():
        raw = output_file.read_bytes()
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", errors="replace")
        pages["01_pytest_terminal_output"] = _terminal_page(text.replace("\r\n", "\n").strip())
    for record in records:
        pages[record["val_id"]] = _record_page(record, env, cfg)

    for name, content in pages.items():
        (html_dir / f"{name}.html").write_text(content, encoding="utf-8")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        tab = browser.new_page(viewport={"width": 1240, "height": 800}, device_scale_factor=1.5)
        for name in pages:
            tab.goto((html_dir / f"{name}.html").resolve().as_uri())
            tab.wait_for_load_state("load")
            tab.screenshot(path=str(shot_dir / f"{name}.png"), full_page=True)
            print(f"screenshot: {shot_dir / f'{name}.png'}")
        browser.close()
    return 0
