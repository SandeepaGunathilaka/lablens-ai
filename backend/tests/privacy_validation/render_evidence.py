"""Render the Student 2 privacy assessment evidence as HTML pages and full-page PNG screenshots.

Usage (from backend/):
    python -m pytest tests/privacy_validation -v -rxXs | Tee-Object tests/privacy_validation/evidence/pytest_output.txt
    python tests/privacy_validation/render_evidence.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evidence_support import render  # noqa: E402

if __name__ == "__main__":
    sys.exit(render(
        Path(__file__).parent,
        title="LabLens AI - Privacy and Data Leakage assessment results (Student 2)",
        subtitle="Test cases PRIV-01 to PRIV-15, executed against the real FastAPI backend and the real React frontend (Edge)",
        planned=15,
        id_regex=r"PRIV-\d+",
        expected_heading="Expected behaviour (assessment section 5)",
        footer=[("Commit", "git_commit"), ("on", "git_branch"), ("Python", "python"), ("pytest", "pytest"),
                ("executed", "executed_at")],
    ))
