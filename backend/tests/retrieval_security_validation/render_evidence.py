"""Render the Student 4 IR assessment evidence as HTML pages and full-page PNG screenshots.

Usage (from backend/):
    python -m pytest tests/retrieval_security_validation -v -rxXs | Tee-Object tests/retrieval_security_validation/evidence/pytest_output.txt
    python tests/retrieval_security_validation/render_evidence.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evidence_support import render  # noqa: E402

if __name__ == "__main__":
    sys.exit(render(
        Path(__file__).parent,
        title="LabLens AI - Information Retrieval and Security assessment results (Student 4)",
        subtitle="Test cases IR-01 to IR-15, executed against the real Medical Retrieval Agent (MiniLM + Chroma) and FastAPI auth/API layer",
        planned=15,
        id_regex=r"IR-\d+",
        expected_heading="Expected result (assessment section 3)",
        footer=[("Commit", "git_commit"), ("on", "git_branch"), ("Python", "python"), ("pytest", "pytest"),
                ("Model", "embedding_model"), ("executed", "executed_at")],
    ))
