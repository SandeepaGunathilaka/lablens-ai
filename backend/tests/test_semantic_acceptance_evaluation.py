"""Offline unit tests for held-out evaluation of the frozen semantic acceptance policy."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents import evaluate_semantic_acceptance as heldout
from agents.evaluate_semantic_retrieval import (
    DATASET_PATH,
    dataset_sha256,
    load_dataset,
)
from agents.knowledge_base import KNOWLEDGE_BASE_DIR
from agents.knowledge_base_fingerprint import DOCUMENT_IDS, knowledge_base_sha256


@pytest.fixture(autouse=True)
def no_real_components(monkeypatch):
    """Ensure no real sentence-transformers, embeddings, or Chroma connections are made."""
    for name in ("EmbeddingService", "ChromaVectorStore", "SemanticRetriever"):
        monkeypatch.setattr(heldout, name, Mock(side_effect=AssertionError("Real runtime forbidden in unit tests")))


def synthetic_heldout_query(
    qid: str,
    doc_id: str | None,
    sim: float | None,
    margin: float | None,
    top1_correct: bool | None = None,
    split: str = "held_out",
    cat: str = "direct_semantic",
    top1_doc_id: str | None = None,
    accepted: bool | None = None,
    outcome: str | None = None,
) -> dict:
    actual_top1_doc = top1_doc_id if top1_doc_id is not None else (doc_id if (top1_correct is True or (top1_correct is None and doc_id is not None)) else "wbc")
    return {
        "id": qid,
        "query": f"synthetic text for {qid}",
        "category": cat if doc_id is not None else "unsupported_medical",
        "expected_document_id": doc_id,
        "split": split,
        "group_id": f"group_{qid}",
        "top1_similarity": sim,
        "similarity_margin": margin,
        "top1_document_id": actual_top1_doc,
        "top1_correct": top1_correct if doc_id is not None else None,
        "top1_distance": 1.0 - sim if sim is not None else None,
        "top2_document_id": "total_cholesterol",
        "top2_similarity": sim - margin if (sim is not None and margin is not None) else None,
        "top2_distance": 1.0 - (sim - margin) if (sim is not None and margin is not None) else None,
        "accepted": accepted,
        "outcome": outcome,
        "ranked_candidates": [],
        "error": None,
    }


def test_frozen_policy_immutability_and_acceptance():
    policy = heldout.FrozenAcceptancePolicy()
    assert policy.similarity_threshold == 0.0
    assert abs(policy.margin_threshold - 0.22541916370391846) < 1e-12

    # Both pass
    assert policy.accepts(similarity=0.1, margin=0.25) is True
    assert policy.accepts(similarity=0.0, margin=0.22541916370391846) is True

    # Similarity fails
    assert policy.accepts(similarity=-0.05, margin=0.30) is False

    # Margin fails
    assert policy.accepts(similarity=0.5, margin=0.20) is False

    # Missing / non-finite scores fail safely
    assert policy.accepts(similarity=None, margin=0.25) is False
    assert policy.accepts(similarity=0.5, margin=None) is False
    assert policy.accepts(similarity=float("nan"), margin=0.25) is False
    assert policy.accepts(similarity=0.5, margin=float("inf")) is False
    assert policy.accepts(similarity=float("inf"), margin=0.25) is False


def test_classify_heldout_outcome():
    # Supported
    assert heldout.classify_heldout_outcome(is_supported=True, top1_correct=True, accepted=True) == "correct_accept"
    assert heldout.classify_heldout_outcome(is_supported=True, top1_correct=False, accepted=True) == "wrong_accept"
    assert heldout.classify_heldout_outcome(is_supported=True, top1_correct=True, accepted=False) == "abstain_correct_candidate"
    assert heldout.classify_heldout_outcome(is_supported=True, top1_correct=False, accepted=False) == "abstain_wrong_candidate"

    # Negative
    assert heldout.classify_heldout_outcome(is_supported=False, top1_correct=None, accepted=True) == "false_accept"
    assert heldout.classify_heldout_outcome(is_supported=False, top1_correct=None, accepted=False) == "correct_abstain"


def test_calculate_heldout_metrics():
    queries = [
        synthetic_heldout_query("s1", "hemoglobin", sim=0.5, margin=0.3, top1_correct=True, accepted=True, outcome="correct_accept"),
        synthetic_heldout_query("s2", "hemoglobin", sim=0.5, margin=0.3, top1_correct=True, accepted=True, outcome="correct_accept"),
        synthetic_heldout_query("s3", "hemoglobin", sim=0.5, margin=0.3, top1_correct=False, accepted=True, outcome="wrong_accept"),
        synthetic_heldout_query("s4", "hemoglobin", sim=0.3, margin=0.1, top1_correct=True, accepted=False, outcome="abstain_correct_candidate"),
        synthetic_heldout_query("n1", None, sim=0.5, margin=0.3, accepted=True, outcome="false_accept"),
        synthetic_heldout_query("n2", None, sim=0.1, margin=0.05, accepted=False, outcome="correct_abstain"),
    ]

    metrics = heldout.calculate_heldout_metrics(queries)
    assert metrics["heldout_total"] == 6
    assert metrics["supported_total"] == 4
    assert metrics["negative_total"] == 2
    assert metrics["accepted_count"] == 4
    assert metrics["abstained_count"] == 2

    assert metrics["correct_accept_count"] == 2
    assert metrics["wrong_accept_count"] == 1
    assert metrics["negative_false_accept_count"] == 1

    # Precision: 2 / 4 = 0.5
    assert metrics["accepted_precision"] == pytest.approx(0.5)
    # Supported correct coverage: 2 / 4 = 0.5
    assert metrics["supported_correct_coverage"] == pytest.approx(0.5)
    # Negative false accept rate: 1 / 2 = 0.5
    assert metrics["negative_false_accept_rate"] == pytest.approx(0.5)
    # Wrong supported accept rate: 1 / 4 = 0.25
    assert metrics["wrong_supported_accept_rate"] == pytest.approx(0.25)
    # Overall abstention rate: 2 / 6 = 1/3
    assert metrics["overall_abstention_rate"] == pytest.approx(1 / 3)


def test_calculate_heldout_metrics_zero_denominators():
    queries = [
        synthetic_heldout_query("s1", "hemoglobin", sim=0.1, margin=0.01, top1_correct=True, accepted=False, outcome="abstain_correct_candidate"),
    ]
    metrics = heldout.calculate_heldout_metrics(queries)
    assert metrics["accepted_count"] == 0
    assert metrics["accepted_precision"] is None
    assert metrics["negative_false_accept_rate"] is None
    assert metrics["overall_abstention_rate"] == 1.0


def test_calculate_per_test_metrics():
    queries = []
    for doc in DOCUMENT_IDS:
        for i in range(4):
            corr = (i < 3)
            acc = (doc == "hemoglobin" and i == 0)
            outcome = "correct_accept" if acc else ("abstain_correct_candidate" if corr else "abstain_wrong_candidate")
            queries.append(synthetic_heldout_query(f"{doc}_{i}", doc, sim=0.5, margin=0.3, top1_correct=corr, accepted=acc, outcome=outcome))

    per_test = heldout.calculate_per_test_metrics(queries)
    assert len(per_test) == 7
    for doc in DOCUMENT_IDS:
        assert per_test[doc]["heldout_supported_queries"] == 4
        assert per_test[doc]["top1_correct"] == 3

    assert per_test["hemoglobin"]["accepted"] == 1
    assert per_test["hemoglobin"]["correct_accept"] == 1
    assert per_test["hemoglobin"]["supported_correct_coverage"] == 0.25
    assert per_test["hemoglobin"]["accepted_precision"] == 1.0

    assert per_test["ldl"]["accepted"] == 0
    assert per_test["ldl"]["correct_accept"] == 0
    assert per_test["ldl"]["supported_correct_coverage"] == 0.0
    assert per_test["ldl"]["accepted_precision"] is None


def test_calculate_category_metrics():
    queries = [
        synthetic_heldout_query("s1", "hemoglobin", sim=0.5, margin=0.3, top1_correct=True, cat="direct_semantic", accepted=True, outcome="correct_accept"),
        synthetic_heldout_query("s2", "hemoglobin", sim=0.2, margin=0.1, top1_correct=True, cat="direct_semantic", accepted=False, outcome="abstain_correct_candidate"),
        synthetic_heldout_query("n1", None, sim=0.5, margin=0.3, cat="unsupported_medical", accepted=True, outcome="false_accept"),
        synthetic_heldout_query("n2", None, sim=0.1, margin=0.05, cat="unsupported_medical", accepted=False, outcome="correct_abstain"),
    ]
    cat_metrics = heldout.calculate_category_metrics(queries)
    assert cat_metrics["direct_semantic"]["total"] == 2
    assert cat_metrics["direct_semantic"]["accepted"] == 1
    assert cat_metrics["direct_semantic"]["correct_accept"] == 1
    assert cat_metrics["direct_semantic"]["supported_correct_coverage"] == 0.5

    assert cat_metrics["unsupported_medical"]["total"] == 2
    assert cat_metrics["unsupported_medical"]["accepted"] == 1
    assert cat_metrics["unsupported_medical"]["false_accept"] == 1
    assert cat_metrics["unsupported_medical"]["false_accept_rate"] == 0.5
    assert cat_metrics["unsupported_medical"]["abstention_rate"] == 0.5


def test_heldout_split_firewall_and_calibration_exclusion():
    # Attempting to pass calibration queries must be rejected by evaluate_heldout_queries
    raw_cal = [
        synthetic_heldout_query("c1", "hemoglobin", sim=0.5, margin=0.2, split="calibration"),
    ]
    dataset = SimpleNamespace(queries=raw_cal)
    with pytest.raises(ValueError, match="Expected exactly 48 held-out queries"):
        heldout.evaluate_heldout_queries(dataset, Mock(), heldout.FrozenAcceptancePolicy())


def test_compare_calibration_vs_heldout():
    cal_m = {
        "accepted_precision": 1.0,
        "supported_correct_coverage": 0.125,
        "negative_false_accept_rate": 0.0,
        "wrong_supported_accept_rate": 0.0,
        "overall_abstention_rate": 0.9271,
    }
    held_m = {
        "accepted_precision": 1.0,
        "supported_correct_coverage": 0.1071,
        "negative_false_accept_rate": 0.0,
        "wrong_supported_accept_rate": 0.0,
        "overall_abstention_rate": 0.9375,
    }
    comp = heldout.compare_calibration_vs_heldout(cal_m, held_m)
    assert comp["accepted_precision"]["calibration"] == 1.0
    assert comp["accepted_precision"]["held_out"] == 1.0
    assert comp["accepted_precision"]["difference"] == 0.0
    assert comp["supported_correct_coverage"]["difference"] == pytest.approx(0.1071 - 0.125)


@pytest.mark.parametrize('calibration_state', ['missing', 'empty', 'null_metrics', 'present'])
def test_build_heldout_report_serialization(tmp_path, monkeypatch, calibration_state):
    monkeypatch.setattr(heldout, "git_revision", lambda: "fake-git-rev")

    # Generate 48 synthetic held-out queries: 28 supported, 20 negative
    rows = []
    for i in range(28):
        doc = DOCUMENT_IDS[i % len(DOCUMENT_IDS)]
        acc = (i == 0)
        outcome = "correct_accept" if acc else "abstain_correct_candidate"
        rows.append(synthetic_heldout_query(f"held_s_{i}", doc, sim=0.5 if acc else 0.2, margin=0.3 if acc else 0.05,
                                           top1_correct=True, accepted=acc, outcome=outcome))
    for i in range(20):
        rows.append(synthetic_heldout_query(f"held_n_{i}", None, sim=0.1, margin=0.02, accepted=False, outcome="correct_abstain"))

    store = SimpleNamespace(
        kb_directory=KNOWLEDGE_BASE_DIR,
        model_name="fake-model",
        model_revision=None,
        dimension=384,
        collection_name="fake-collection",
    )

    cal_path = tmp_path / "calibration.json"
    if calibration_state != "missing":
        metrics = ({"accepted_precision": 1.0, "supported_correct_coverage": 0.125}
                   if calibration_state == "present" else
                   {"accepted_precision": None, "supported_correct_coverage": None}
                   if calibration_state == "null_metrics" else {})
        cal_path.write_text(json.dumps({"selected_policy": {"calibration_metrics": metrics}}), encoding="utf-8")
    report = heldout.build_heldout_report(store=store, rows=rows, calibration_report_path=cal_path)
    summary = report["generalization_summary"]
    assert summary["precision_status"] == ("retained" if calibration_state == "present" else "unavailable")
    assert summary["coverage_status"] == ("reduced" if calibration_state == "present" else "unavailable")

    assert report["evaluation"]["name"] == "semantic_acceptance_heldout"
    assert report["evaluation"]["dataset_sha256"] == dataset_sha256()
    assert report["evaluation"]["kb_sha256"] == knowledge_base_sha256()
    assert report["evaluation"]["heldout_query_count"] == 48
    assert report["evaluation"]["heldout_supported_count"] == 28
    assert report["evaluation"]["heldout_negative_count"] == 20

    assert report["frozen_policy"]["similarity_threshold"] == 0.0
    assert abs(report["frozen_policy"]["margin_threshold"] - 0.22541916370391846) < 1e-12

    assert len(report["queries"]) == 48
    assert "heldout_metrics" in report
    assert "per_test_metrics" in report
    assert "category_metrics" in report
    assert "generalization_summary" in report

    out_file = tmp_path / "heldout_report.json"
    heldout.write_heldout_report(report, out_file)
    assert out_file.exists()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["evaluation"]["name"] == "semantic_acceptance_heldout"


def test_production_benchmark_hashes():
    assert dataset_sha256() == "cf799511635cb44add34e8710933142a45a1c717a8f712551f7f42809b1dd0e1"
    assert knowledge_base_sha256() == "f704ff3d180124a90638695413f594531e7609e0a0902257dcd36b10a7c4bcc4"


def test_cli_runner(monkeypatch, capsys):
    mock_report = {
        "frozen_policy": {
            "similarity_threshold": 0.0,
            "margin_threshold": 0.2254,
        },
        "heldout_metrics": {
            "heldout_total": 48,
            "supported_total": 28,
            "negative_total": 20,
            "accepted_count": 3,
            "abstained_count": 45,
            "correct_accept_count": 3,
            "wrong_accept_count": 0,
            "negative_false_accept_count": 0,
            "accepted_precision": 1.0,
            "supported_correct_coverage": 0.1071,
            "negative_false_accept_rate": 0.0,
            "wrong_supported_accept_rate": 0.0,
            "overall_abstention_rate": 0.9375,
        }
    }
    monkeypatch.setattr(heldout, "build_heldout_report", lambda: mock_report)
    monkeypatch.setattr(heldout, "write_heldout_report", lambda report: None)

    exit_code = heldout.main()
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "similarity >= 0.0" in out
    assert "margin >= 0.2254" in out
    assert "Held-out Accepted: 3/48" in out
