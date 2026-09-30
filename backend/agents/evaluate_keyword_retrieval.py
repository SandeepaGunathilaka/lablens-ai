"""Reproducible offline evaluation of the exact-match retrieval contract."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from agents.knowledge_base import KNOWLEDGE_BASE_DIR, KnowledgeBaseLoader
from agents.keyword_retriever import KeywordRetriever
from agents.knowledge_base_fingerprint import DOCUMENT_IDS, knowledge_base_sha256


BACKEND_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BACKEND_DIR / "tests" / "fixtures" / "retrieval" / "keyword_queries.json"
REPORT_PATH = BACKEND_DIR / "evaluation_results" / "keyword_baseline.json"
CATEGORY_COUNTS = {"canonical": 7, "alias": 26, "normalized_variant": 21, "unsupported": 14}
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EvaluationQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: NonBlank
    query: str  # Preserve blanks and formatting exactly.
    category: Literal["canonical", "alias", "normalized_variant", "unsupported"]
    expected_document_id: str | None

    @model_validator(mode="after")
    def validate_expected_id(self) -> "EvaluationQuery":
        if self.category == "unsupported":
            if self.expected_document_id is not None:
                raise ValueError("Unsupported cases require a null expected_document_id")
        elif self.expected_document_id not in DOCUMENT_IDS:
            raise ValueError("Supported cases require one of the seven document IDs")
        return self


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version: NonBlank
    description: NonBlank
    queries: list[EvaluationQuery] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> "EvaluationDataset":
        ids = [query.id for query in self.queries]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate query IDs")
        return self


def load_dataset(path: Path | str = DATASET_PATH, *, production: bool = True) -> EvaluationDataset:
    dataset = EvaluationDataset.model_validate_json(Path(path).read_bytes())
    if production and Counter(q.category for q in dataset.queries) != CATEGORY_COUNTS:
        raise ValueError("Production fixture requires exactly 68 queries with category counts 7/26/21/14")
    return dataset


def evaluate_queries(dataset: EvaluationDataset, retriever: KeywordRetriever) -> list[dict]:
    results = []
    for query in dataset.queries:
        actual = None
        error = None
        try:
            document = retriever.retrieve(query.query)
            actual = document.id if document is not None else None
        except Exception as exc:  # Evaluation records failures and continues through every case.
            error = f"{type(exc).__name__}: {exc}"
        results.append({
            **query.model_dump(), "actual_document_id": actual,
            "correct": error is None and actual == query.expected_document_id,
            "error": error,
        })
    return results


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _group_metrics(rows: list[dict]) -> dict:
    correct = sum(row["correct"] for row in rows)
    return {"total": len(rows), "correct": correct, "accuracy": _ratio(correct, len(rows))}


def calculate_metrics(results: list[dict]) -> dict:
    supported = [row for row in results if row["expected_document_id"] is not None]
    unsupported = [row for row in results if row["expected_document_id"] is None]
    correct_supported = sum(row["correct"] for row in supported)
    correct_rejections = sum(row["correct"] for row in unsupported)
    answered = sum(row["actual_document_id"] is not None for row in results)
    supported_answered = sum(row["actual_document_id"] is not None for row in supported)
    correct_total = correct_supported + correct_rejections
    categories = {
        category: _group_metrics([row for row in results if row["category"] == category])
        for category in CATEGORY_COUNTS
    }
    per_test = {
        identifier: _group_metrics([row for row in supported if row["expected_document_id"] == identifier])
        for identifier in DOCUMENT_IDS
    }
    # Small synthetic datasets may omit tests; absent groups have null accuracy.
    accuracies = [group["accuracy"] for group in per_test.values() if group["total"]]
    return {
        "counts": {
            "total_queries": len(results), "supported_queries": len(supported),
            "unsupported_queries": len(unsupported), "correct_supported": correct_supported,
            "correct_unsupported_rejections": correct_rejections, "correct_total": correct_total,
            "answered_queries": answered, "supported_answered_queries": supported_answered,
            "error_queries": sum(row["error"] is not None for row in results),
        },
        "metrics": {
            "overall_accuracy": _ratio(correct_total, len(results)),
            "supported_top1_accuracy": _ratio(correct_supported, len(supported)),
            "unsupported_rejection_accuracy": _ratio(correct_rejections, len(unsupported)),
            "answer_coverage": _ratio(answered, len(results)),
            "supported_query_coverage": _ratio(supported_answered, len(supported)),
            "precision_among_answered_queries": _ratio(correct_supported, answered),
            "macro_supported_test_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
        },
        "category_metrics": categories,
        "per_test_metrics": per_test,
    }


def dataset_sha256(path: Path | str = DATASET_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=BACKEND_DIR.parent,
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_report(
    dataset_path: Path | str = DATASET_PATH,
    kb_directory: Path | str = KNOWLEDGE_BASE_DIR,
) -> dict:
    dataset_hash = dataset_sha256(dataset_path)
    kb_hash = knowledge_base_sha256(kb_directory)
    dataset = load_dataset(dataset_path)
    retriever = KeywordRetriever(KnowledgeBaseLoader(kb_directory))
    results = evaluate_queries(dataset, retriever)
    if dataset_sha256(dataset_path) != dataset_hash or knowledge_base_sha256(kb_directory) != kb_hash:
        raise ValueError("Evaluation inputs changed during execution; rerun with stable files")
    return {
        "evaluation": {
            "name": "keyword_retrieval_baseline", "dataset_version": dataset.dataset_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_revision": git_revision(), "dataset_sha256": dataset_hash,
            "knowledge_base_sha256": kb_hash, "python_version": platform.python_version(),
        },
        **calculate_metrics(results), "queries": results,
    }


def write_report(report: dict, path: Path | str = REPORT_PATH) -> None:
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def main() -> int:
    try:
        report = build_report()
        write_report(report)
    except (OSError, ValueError) as exc:
        print(f"Evaluation failed: {exc}")
        return 1
    counts = report["counts"]
    print(f"Keyword baseline: {counts['correct_total']}/{counts['total_queries']} correct; "
          f"{counts['error_queries']} retrieval errors")
    print(f"Overall accuracy: {report['metrics']['overall_accuracy']:.2%}")
    print(f"Report: {REPORT_PATH}")
    return 1 if counts["error_queries"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
