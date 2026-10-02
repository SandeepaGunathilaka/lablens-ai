"""Render the Responsible AI evaluation evidence (group report section 7.2) as HTML pages and PNG screenshots.

Usage (from backend/):
    python -m pytest tests/responsible_ai_evaluation -v -rxXs | Tee-Object tests/responsible_ai_evaluation/evidence/pytest_output.txt
    python tests/responsible_ai_evaluation/render_evidence.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evidence_support import render  # noqa: E402

if __name__ == "__main__":
    sys.exit(render(
        Path(__file__).parent,
        title="LabLens AI - Responsible AI evaluation results (RA-01 to RA-15)",
        subtitle="Executed end to end over HTTP: auth, upload, Document Agent, hybrid retrieval, Explanation Agent, "
                 "Safety Agent and follow-up chat",
        planned=15,
        id_regex=r"RA-\d+",
        expected_heading="Expected result (report section 7.2)",
        footer=[("Commit", "git_commit"), ("on", "git_branch"), ("Python", "python"), ("pytest", "pytest"),
                ("executed", "executed_at")],
    ))
