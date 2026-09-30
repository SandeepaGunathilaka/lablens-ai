"""Fixed-label evaluation and independent checks of metric arithmetic."""

from collections import Counter
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents import evaluate_keyword_retrieval as evaluation
from agents.knowledge_base import KnowledgeBaseLoader
from agents.keyword_retriever import KeywordRetriever


def query(identifier="q1", category="canonical", expected="hemoglobin", text="Hemoglobin"):
    return {"id": identifier, "query": text, "category": category, "expected_document_id": expected}


def dataset(queries):
    return {"dataset_version": "1.0", "description": "Synthetic evaluation", "queries": queries}


def save_dataset(tmp_path, payload):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_dataset_preserves_query_whitespace(tmp_path):
    path = save_dataset(tmp_path, dataset([query(text="  Hemoglobin  ")]))
    loaded = evaluation.load_dataset(path, production=False)
    assert loaded.queries[0].query == "  Hemoglobin  "


@pytest.mark.parametrize("payload", [
    {}, [], {"dataset_version": "", "description": "x", "queries": [query()]},
    {"dataset_version": " ", "description": "x", "queries": [query()]},
    dataset([]), dataset("not a list"), dataset([{}]), dataset([query(), query()]),
    dataset([query(category="invalid")]), dataset([query(expected="unknown")]),
    dataset([query(expected=None)]), dataset([query(category="unsupported")]),
    dataset([query(text=None)]), dataset([query(identifier=" ")]),
    dataset([{key: value for key, value in query().items() if key != "expected_document_id"}]),
    dataset([{key: value for key, value in query().items() if key != "query"}]),
])
def test_rejects_malformed_dataset(tmp_path, payload):
    with pytest.raises(ValueError):
        evaluation.load_dataset(save_dataset(tmp_path, payload), production=False)


def test_invalid_json_is_not_skipped(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        evaluation.load_dataset(path)


def test_production_counts_are_enforced(tmp_path):
    with pytest.raises(ValueError, match="68"):
        evaluation.load_dataset(save_dataset(tmp_path, dataset([query()])))
    fixture = evaluation.load_dataset().model_dump()
    fixture["queries"][0]["category"] = "alias"
    with pytest.raises(ValueError, match="category counts"):
        evaluation.load_dataset(save_dataset(tmp_path, fixture))


def test_fixed_fixture_labels_and_category_counts():
    fixture = evaluation.load_dataset()
    assert len(fixture.queries) == 68
    assert Counter(q.category for q in fixture.queries) == evaluation.CATEGORY_COUNTS
    documents = KnowledgeBaseLoader().load_all()
    assert {(q.query, q.expected_document_id) for q in fixture.queries if q.category == "canonical"} == {
        (doc.test_name, doc.id) for doc in documents
    }
    assert {(q.query, q.expected_document_id) for q in fixture.queries if q.category == "alias"} == {
        (alias, doc.id) for doc in documents for alias in doc.aliases
    }
    variants = Counter(q.expected_document_id for q in fixture.queries if q.category == "normalized_variant")
    assert variants == {identifier: 3 for identifier in evaluation.DOCUMENT_IDS}
    assert len({q.query for q in fixture.queries}) == 68


def test_mixed_outcomes_metrics_categories_and_macro_average():
    fixture = evaluation.EvaluationDataset.model_validate(dataset([
        query("a"), query("b", "alias"), query("c", "normalized_variant", "wbc"),
        query("d", "unsupported", None), query("e", "unsupported", None),
        query("f", "unsupported", None),
    ]))
    retriever = Mock()
    retriever.retrieve.side_effect = [
        SimpleNamespace(id="hemoglobin"), SimpleNamespace(id="ldl"), None,
        None, SimpleNamespace(id="hdl"), RuntimeError("broken"),
    ]
    rows = evaluation.evaluate_queries(fixture, retriever)
    report = evaluation.calculate_metrics(rows)
    assert report["counts"] == {
        "total_queries": 6, "supported_queries": 3, "unsupported_queries": 3,
        "correct_supported": 1, "correct_unsupported_rejections": 1, "correct_total": 2,
        "answered_queries": 3, "supported_answered_queries": 2, "error_queries": 1,
    }
    assert report["metrics"] == {
        "overall_accuracy": 2 / 6, "supported_top1_accuracy": 1 / 3,
        "unsupported_rejection_accuracy": 1 / 3, "answer_coverage": 3 / 6,
        "supported_query_coverage": 2 / 3, "precision_among_answered_queries": 1 / 3,
        "macro_supported_test_accuracy": 0.25,
    }
    assert report["category_metrics"] == {
        "canonical": {"total": 1, "correct": 1, "accuracy": 1.0},
        "alias": {"total": 1, "correct": 0, "accuracy": 0.0},
        "normalized_variant": {"total": 1, "correct": 0, "accuracy": 0.0},
        "unsupported": {"total": 3, "correct": 1, "accuracy": 1 / 3},
    }
    assert report["per_test_metrics"]["hemoglobin"] == {"total": 2, "correct": 1, "accuracy": 0.5}
    assert report["per_test_metrics"]["wbc"] == {"total": 1, "correct": 0, "accuracy": 0.0}
    assert report["per_test_metrics"]["hdl"] == {"total": 0, "correct": 0, "accuracy": None}
    assert rows[-1]["correct"] is False
    assert rows[-1]["error"] == "RuntimeError: broken"
    assert rows[3]["correct"] is True and rows[3]["error"] is None


def test_zero_denominators():
    assert all(value is None for value in evaluation.calculate_metrics([])["metrics"].values())
    fixture = evaluation.EvaluationDataset.model_validate(dataset([query(category="unsupported", expected=None)]))
    retriever = Mock()
    retriever.retrieve.return_value = None
    metrics = evaluation.calculate_metrics(evaluation.evaluate_queries(fixture, retriever))["metrics"]
    assert metrics["unsupported_rejection_accuracy"] == 1
    assert metrics["precision_among_answered_queries"] is None
    assert metrics["supported_top1_accuracy"] is None
    assert metrics["supported_query_coverage"] is None
    assert metrics["macro_supported_test_accuracy"] is None


def test_dataset_hash_uses_exact_bytes(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_bytes(b'{"x": 1}\n')
    expected = hashlib.sha256(b'{"x": 1}\n').hexdigest()
    assert evaluation.dataset_sha256(path) == expected
    assert evaluation.dataset_sha256(path) == expected
    path.write_bytes(b'{"x":1}\n')
    assert evaluation.dataset_sha256(path) != expected


def test_kb_hash_is_sorted_and_sensitive_to_raw_bytes(tmp_path):
    expected = hashlib.sha256()
    for identifier in reversed(evaluation.DOCUMENT_IDS):
        (tmp_path / f"{identifier}.json").write_bytes(b"{}")
    for filename in sorted(f"{identifier}.json" for identifier in evaluation.DOCUMENT_IDS):
        for raw in [filename.encode("utf-8"), b"{}"]:
            expected.update(len(raw).to_bytes(8, "big"))
            expected.update(raw)
    digest = evaluation.knowledge_base_sha256(tmp_path)
    assert digest == expected.hexdigest()
    assert evaluation.knowledge_base_sha256(tmp_path) == digest
    (tmp_path / "README.md").write_text("ignored", encoding="utf-8")
    assert evaluation.knowledge_base_sha256(tmp_path) == digest
    (tmp_path / "hdl.json").write_bytes(b"{}\n")
    assert evaluation.knowledge_base_sha256(tmp_path) != digest


def test_kb_hash_rejects_incomplete_directory(tmp_path):
    with pytest.raises(ValueError, match="seven"):
        evaluation.knowledge_base_sha256(tmp_path)


def test_git_unavailable_returns_none(monkeypatch):
    monkeypatch.setattr(evaluation.subprocess, "run", Mock(side_effect=OSError("missing git")))
    assert evaluation.git_revision() is None


def test_git_revision_success(monkeypatch):
    monkeypatch.setattr(evaluation.subprocess, "run", Mock(return_value=SimpleNamespace(stdout="abc123\n")))
    assert evaluation.git_revision() == "abc123"


def test_complete_baseline_report_and_output_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "git_revision", lambda: None)
    report = evaluation.build_report()
    assert report["counts"]["total_queries"] == 68
    assert report["counts"]["correct_total"] == 68
    assert report["counts"]["answered_queries"] == 54
    assert report["counts"]["error_queries"] == 0
    assert report["metrics"]["overall_accuracy"] == 1
    assert report["metrics"]["answer_coverage"] == 54 / 68
    assert report["metrics"]["macro_supported_test_accuracy"] == 1
    assert all(group["accuracy"] == 1 for group in report["per_test_metrics"].values())
    assert report["evaluation"]["dataset_sha256"] == evaluation.dataset_sha256()
    assert report["evaluation"]["knowledge_base_sha256"] == evaluation.knowledge_base_sha256()
    assert report["evaluation"]["git_revision"] is None
    path = tmp_path / "new" / "nested" / "report.json"
    evaluation.write_report(report, path)
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_cli_writes_report_and_summary(tmp_path, monkeypatch, capsys):
    report = {"counts": {"correct_total": 68, "total_queries": 68, "error_queries": 0},
              "metrics": {"overall_accuracy": 1}}
    monkeypatch.setattr(evaluation, "build_report", lambda: report)
    writer = Mock()
    monkeypatch.setattr(evaluation, "write_report", writer)
    assert evaluation.main() == 0
    writer.assert_called_once_with(report)
    assert "68/68" in capsys.readouterr().out


def test_cli_retrieval_errors_return_failure(monkeypatch):
    monkeypatch.setattr(evaluation, "build_report", lambda: {
        "counts": {"correct_total": 0, "total_queries": 1, "error_queries": 1},
        "metrics": {"overall_accuracy": 0},
    })
    monkeypatch.setattr(evaluation, "write_report", Mock())
    assert evaluation.main() == 1
