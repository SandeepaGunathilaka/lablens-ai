"""Render the validation evidence as HTML pages and full-page PNG screenshots.

Usage (from backend/):
    python -m pytest tests/explanation_validation -v -rxX | Tee-Object tests/explanation_validation/evidence/pytest_output.txt
    python tests/explanation_validation/render_evidence.py

Reads evidence/results.json (written by conftest.py) and evidence/pytest_output.txt, writes
evidence/html/*.html and evidence/screenshots/*.png using the installed Microsoft Edge.
"""

import html
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

EVIDENCE_DIR = Path(__file__).parent / "evidence"
HTML_DIR = EVIDENCE_DIR / "html"
SHOT_DIR = EVIDENCE_DIR / "screenshots"

STATUS = {
    "passed": ("PASS", "pass"),
    "xfailed": ("FAIL - confirmed finding", "fail"),
    "failed": ("FAIL", "fail"),
    "xpassed": ("PASS (finding no longer reproduces)", "pass"),
}

CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 28px; background: #f4f6fa; color: #1d2433; width: 1180px; }
.card { background: #fff; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.08); padding: 24px 28px; }
h1 { margin: 0 0 4px; font-size: 24px; } h2 { font-size: 16px; margin: 22px 0 8px; color: #33415c; }
.meta { color: #5c677d; font-size: 13px; margin-bottom: 4px; }
.badge { display: inline-block; padding: 5px 14px; border-radius: 999px; font-weight: 700; font-size: 14px; color: #fff; }
.badge.pass { background: #1b9e5a; } .badge.fail { background: #d64545; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { background: #24324d; color: #fff; text-align: left; padding: 8px 10px; }
td { border-bottom: 1px solid #e3e7ef; padding: 7px 10px; vertical-align: top; word-break: break-word; }
td:first-child { white-space: nowrap; }
tr.ok td.res { color: #1b9e5a; font-weight: 700; } tr.bad td.res { color: #d64545; font-weight: 700; }
tr.bad { background: #fff1f1; }
td.exp { background: #eef5ff; } td.act { background: #f6fff4; } tr.bad td.act { background: #ffe3e3; }
pre { background: #0f1724; color: #d9e2f2; padding: 12px 14px; border-radius: 8px; font-size: 12px; white-space: pre-wrap; word-break: break-word; margin: 6px 0; }
.kv td:first-child { width: 230px; font-weight: 600; color: #33415c; }
.note { background: #fff8e1; border-left: 4px solid #f0b400; padding: 8px 12px; font-size: 13px; margin-top: 10px; }
.acc { background: #f0f3f9; border-left: 4px solid #24324d; padding: 8px 12px; font-size: 13px; }
"""


def esc(value) -> str:
    return html.escape(str(value))


def page(title: str, body: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{esc(title)}</title><style>{CSS}</style></head><body>{body}</body></html>"


def env_footer(env: dict) -> str:
    return (
        f"<div class='meta' style='margin-top:18px'>Commit {esc(env['git_commit'][:10])} on {esc(env['git_branch'])} | "
        f"Python {esc(env['python'])} | pytest {esc(env['pytest'])} | prompt sha256 {esc(env['prompt_sha256'])} | "
        f"KB sha256 {esc(env['knowledge_base_sha256'])} | executed {esc(env['executed_at'])}</div>"
    )


def record_page(record: dict, env: dict) -> str:
    label, css = STATUS.get(record["outcome"], (record["outcome"].upper(), "fail"))
    if record.get("finding") and record["outcome"] == "xfailed":
        label = f"FAIL - confirmed finding {record['finding']}"
    elif record.get("finding") and record["outcome"] == "passed":
        label = f"PASS - retest of {record['finding']} (fixed)"
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
    body = f"""
<div class='card'>
  <div style='display:flex;justify-content:space-between;align-items:center'>
    <div><h1>{esc(record['val_id'])} - {esc(record['title'])}</h1>
    <div class='meta'>Control under test: {esc(record['control'])}</div>
    <div class='meta'>Test: {esc(record['nodeid'])}</div></div>
    <div><span class='badge {css}'>{esc(label)}</span>
    <div class='meta' style='text-align:right;margin-top:6px'>{passed}/{len(record['checks'])} checks met</div></div>
  </div>
  <h2>Acceptance criteria (Report 3)</h2><div class='acc'>{esc(record['acceptance'])}</div>
  <h2>Test input</h2><table class='kv'>{inputs}</table>
  <h2>Expected result vs actual result</h2>
  <table><tr><th>#</th><th>Check</th><th>Expected result</th><th>Actual result</th><th>Result</th></tr>{rows}</table>
  {outputs}{notes}
  {env_footer(env)}
</div>"""
    return page(record["val_id"], body)


def base_id(val_id: str) -> str:
    return re.match(r"VAL-\d+", val_id).group(0)


def retest_table(records: list[dict]) -> str:
    """First-run vs retest outcome for every finding, if first-run evidence was kept."""
    first_file = EVIDENCE_DIR / "first_run" / "results.json"
    if not first_file.exists():
        return ""
    first = {r["val_id"]: r for r in json.loads(first_file.read_text(encoding="utf-8"))["records"]}
    rows = ""
    for record in records:
        before = first.get(record["val_id"])
        if not record.get("finding") or before is None:
            continue
        failed = [c for c in before["checks"] if not c["passed"]]
        original = "; ".join(f"{c['description']}: {c['actual']}" for c in failed) or "passed"
        fixed = record["outcome"] in ("passed", "xpassed")
        rows += (
            f"<tr class='{'ok' if fixed else 'bad'}'><td>{esc(record['finding'])}</td><td>{esc(record['val_id'])}</td>"
            f"<td>FAIL ({len(before['checks']) - len(failed)}/{len(before['checks'])}): {esc(original)}</td>"
            f"<td class='res'>{'PASS' if fixed else 'FAIL'} "
            f"({sum(c['passed'] for c in record['checks'])}/{len(record['checks'])})</td>"
            f"<td>{esc(FIXES.get(record['finding'], ''))}</td></tr>"
        )
    if not rows:
        return ""
    return (
        "<h2>Retesting (Report 3 section 9)</h2><table><tr><th>Finding</th><th>Test</th>"
        "<th>Original result (first run)</th><th>Retest</th><th>Fix</th></tr>" + rows + "</table>"
    )


FIXES = {
    "F-01.1": "explanation_agent/service.py + copy.py: conflicting values for one test are flagged and not interpreted",
    "F-01.2": "main.py: JSON-safe RequestValidationError handler, so NaN/Infinity get a 422 instead of a 500",
    "F-01.3": "coordinator.py: physiological plausibility limits; impossible values get no status and a warning",
}


def summary_page(records: list[dict], env: dict) -> str:
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for record in records:
        groups.setdefault(base_id(record["val_id"]), []).append(record)

    def ok(r):
        return r["outcome"] in ("passed", "xpassed")

    executed = len(groups)
    passed = sum(all(ok(r) for r in rs) for rs in groups.values())
    failed = executed - passed
    findings = sorted({r["finding"] for r in records if r.get("finding") and not ok(r)})
    retested = sorted({r["finding"] for r in records if r.get("finding") and ok(r)})

    def finding_cell(r):
        if not r.get("finding"):
            return ""
        return f"{r['finding']} (fixed)" if ok(r) else r["finding"]

    rows = "".join(
        f"<tr class='{'ok' if ok(r) else 'bad'}'><td>{esc(r['val_id'])}</td><td>{esc(r['title'])}</td>"
        f"<td>{esc(r['acceptance'])}</td><td>{sum(c['passed'] for c in r['checks'])}/{len(r['checks'])}</td>"
        f"<td class='res'>{esc(STATUS.get(r['outcome'], (r['outcome'],))[0].split(' -')[0])}</td>"
        f"<td>{esc(finding_cell(r))}</td></tr>"
        for r in records
    )
    env_rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in env.items())
    body = f"""
<div class='card'>
  <h1>LabLens AI - Explanation Agent validation results (Report 3)</h1>
  <div class='meta'>Validation matrix VAL-01 to VAL-15, executed as automated unit/integration tests</div>
  <h2>Results summary</h2>
  <table class='kv'>
    <tr><td>Total tests planned</td><td>15</td></tr>
    <tr><td>Tests executed</td><td>{executed} ({len(records)} test functions incl. sub-cases)</td></tr>
    <tr><td>Passed</td><td>{passed}</td></tr>
    <tr><td>Failed</td><td>{failed}</td></tr>
    <tr><td>Inconclusive / blocked</td><td>0</td></tr>
    <tr><td>Overall pass rate</td><td>{passed}/{executed} = {100 * passed / executed:.1f}%</td></tr>
    <tr><td>Outstanding findings</td><td>{esc(', '.join(findings) or 'None')}</td></tr>
    <tr><td>Retested after fixes</td><td>{len(retested)}: {esc(', '.join(retested) or 'None')}</td></tr>
  </table>
  <h2>Validation matrix</h2>
  <table><tr><th>ID</th><th>Control</th><th>Acceptance criteria</th><th>Checks met</th><th>Status</th><th>Finding</th></tr>{rows}</table>
  {retest_table(records)}
  <h2>Test environment</h2><table class='kv'>{env_rows}</table>
</div>"""
    return page("Summary", body)


def terminal_page(output: str) -> str:
    colored = esc(output)
    colored = re.sub(r"\b(PASSED)\b", r"<span style='color:#4ade80'>\1</span>", colored)
    colored = re.sub(r"\b(XFAIL|FAILED|ERROR)\b", r"<span style='color:#f87171'>\1</span>", colored)
    body = f"<div style='background:#0c0c0c;border-radius:8px;padding:16px'><div style='color:#9ca3af;font-size:12px;margin-bottom:8px'>PowerShell - backend</div><pre style='background:#0c0c0c;font-family:Consolas,monospace;font-size:12.5px'>{colored}</pre></div>"
    return page("pytest output", body)


def main() -> int:
    data = json.loads((EVIDENCE_DIR / "results.json").read_text(encoding="utf-8"))
    env, records = data["environment"], data["records"]
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    SHOT_DIR.mkdir(parents=True, exist_ok=True)

    pages = {"00_summary": summary_page(records, env)}
    output_file = EVIDENCE_DIR / "pytest_output.txt"
    if output_file.exists():
        raw = output_file.read_bytes()
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", errors="replace")
        pages["01_pytest_terminal_output"] = terminal_page(text.replace("\r\n", "\n").strip())
    for record in records:
        pages[record["val_id"]] = record_page(record, env)

    for name, content in pages.items():
        (HTML_DIR / f"{name}.html").write_text(content, encoding="utf-8")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        tab = browser.new_page(viewport={"width": 1240, "height": 800}, device_scale_factor=1.5)
        for name in pages:
            tab.goto((HTML_DIR / f"{name}.html").resolve().as_uri())
            tab.screenshot(path=str(SHOT_DIR / f"{name}.png"), full_page=True)
            print(f"screenshot: {SHOT_DIR / f'{name}.png'}")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
