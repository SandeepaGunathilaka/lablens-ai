"""Tests for semantic retrieval acceptance-policy calibration using synthetic scores only."""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents import calibrate_semantic_retrieval as calib
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
        monkeypatch.setattr(calib, name, Mock(side_effect=AssertionError("Real runtime forbidden in unit tests")))


def synthetic_query(
    qid: str,
    doc_id: str | None,
    sim: float | None,
    margin: float | None,
    top1_correct: bool | None = None,
    split: str = "calibration",
    cat: str = "direct_semantic",
) -> dict:
    return {
        "id": qid,
        "query": f"synthetic text for {qid}",
        "category": cat if doc_id is not None else "unsupported_medical",
        "expected_document_id": doc_id,
        "split": split,
        "group_id": f"group_{qid}",
        "top1_similarity": sim,
        "similarity_margin": margin,
        "top1_document_id": doc_id if (top1_correct is True or (top1_correct is None and doc_id is not None)) else "wbc",
        "top1_correct": top1_correct if doc_id is not None else None,
        "top1_distance": 1.0 - sim if sim is not None else None,
        "error": None,
    }


def test_classify_outcome():
    # Supported queries
    assert calib.classify_outcome(is_supported=True, top1_correct=True, accepted=True) == "correct_accept"
    assert calib.classify_outcome(is_supported=True, top1_correct=False, accepted=True) == "wrong_accept"
    assert calib.classify_outcome(is_supported=True, top1_correct=True, accepted=False) == "abstain_correct_candidate"
    assert calib.classify_outcome(is_supported=True, top1_correct=False, accepted=False) == "abstain_wrong_candidate"

    # Negative queries
    assert calib.classify_outcome(is_supported=False, top1_correct=None, accepted=True) == "false_accept"
    assert calib.classify_outcome(is_supported=False, top1_correct=None, accepted=False) == "correct_abstain"


def test_calculate_policy_metrics_basic_accounting():
    queries = [
        synthetic_query("s1", "hemoglobin", sim=0.5, margin=0.2, top1_correct=True),   # correct accept
        synthetic_query("s2", "hemoglobin", sim=0.5, margin=0.2, top1_correct=True),   # correct accept
        synthetic_query("s3", "hemoglobin", sim=0.5, margin=0.2, top1_correct=False),  # wrong accept
        synthetic_query("s4", "hemoglobin", sim=0.2, margin=0.05, top1_correct=True),  # abstain correct
        synthetic_query("s5", "hemoglobin", sim=0.1, margin=0.01, top1_correct=False), # abstain wrong
        synthetic_query("n1", None, sim=0.5, margin=0.2),                               # negative false accept
        synthetic_query("n2", None, sim=0.1, margin=0.05),                              # correct abstain
    ]
    # Thresholds: s=0.4, m=0.15
    metrics = calib.calculate_policy_metrics(queries, similarity_threshold=0.4, margin_threshold=0.15)

    assert metrics["supported_total"] == 5
    assert metrics["negative_total"] == 2
    assert metrics["total"] == 7

    assert metrics["correct_accept_count"] == 2
    assert metrics["wrong_accept_count"] == 1
    assert metrics["negative_false_accept_count"] == 1
    assert metrics["accepted_count"] == 4
    assert metrics["abstained_count"] == 3

    assert metrics["abstain_correct_candidate_count"] == 1
    assert metrics["abstain_wrong_candidate_count"] == 1
    assert metrics["correct_abstain_count"] == 1

    # Precision: 2 / 4 = 0.5
    assert metrics["accepted_precision"] == pytest.approx(2 / 4)
    # Coverage: 2 / 5 = 0.4
    assert metrics["supported_correct_coverage"] == pytest.approx(2 / 5)
    # Negative false accept rate: 1 / 2 = 0.5
    assert metrics["negative_false_accept_rate"] == pytest.approx(1 / 2)
    # Wrong supported accept rate: 1 / 5 = 0.2
    assert metrics["wrong_supported_accept_rate"] == pytest.approx(1 / 5)
    # Overall abstention rate: 3 / 7
    assert metrics["overall_abstention_rate"] == pytest.approx(3 / 7)
    # Supported accept rate: 3 / 5 = 0.6
    assert metrics["supported_accept_rate"] == pytest.approx(3 / 5)
    # Negative abstention rate: 1 / 2 = 0.5
    assert metrics["negative_abstention_rate"] == pytest.approx(1 / 2)


def test_calculate_policy_metrics_conjunctive_threshold_logic():
    # Only passes if BOTH similarity >= threshold AND margin >= threshold
    q = [synthetic_query("s1", "hemoglobin", sim=0.4, margin=0.1, top1_correct=True)]

    # Both pass
    m1 = calib.calculate_policy_metrics(q, similarity_threshold=0.3, margin_threshold=0.05)
    assert m1["accepted_count"] == 1

    # Similarity fails
    m2 = calib.calculate_policy_metrics(q, similarity_threshold=0.5, margin_threshold=0.05)
    assert m2["accepted_count"] == 0
    assert m2["abstain_correct_candidate_count"] == 1

    # Margin fails
    m3 = calib.calculate_policy_metrics(q, similarity_threshold=0.3, margin_threshold=0.15)
    assert m3["accepted_count"] == 0
    assert m3["abstain_correct_candidate_count"] == 1


def test_calculate_policy_metrics_zero_denominators():
    # No accepted queries
    queries = [synthetic_query("s1", "hemoglobin", sim=0.1, margin=0.01, top1_correct=True)]
    m = calib.calculate_policy_metrics(queries, similarity_threshold=0.5, margin_threshold=0.5)

    assert m["accepted_count"] == 0
    assert m["accepted_precision"] is None
    assert m["supported_correct_coverage"] == 0.0
    assert m["overall_abstention_rate"] == 1.0


def test_generate_threshold_candidates():
    queries = [
        {"top1_similarity": 0.25, "similarity_margin": 0.05},
        {"top1_similarity": 0.35, "similarity_margin": 0.15},
        {"top1_similarity": None, "similarity_margin": 0.10},  # None ignored
        {"top1_similarity": 0.25, "similarity_margin": float("nan")},  # nan ignored
    ]
    sims, margins = calib.generate_threshold_candidates(queries)

    # Boundaries included
    assert -1.0 in sims
    assert 0.0 in sims
    assert 1.0 in sims
    assert 0.25 in sims
    assert 0.35 in sims
    assert sims == sorted(sims)

    assert 0.0 in margins
    assert 1.0 in margins
    assert 0.05 in margins
    assert 0.15 in margins
    assert margins == sorted(margins)


def test_threshold_precision_and_simplicity():
    assert calib.threshold_precision(0.0) == 0
    assert calib.threshold_precision(1.0) == 0
    assert calib.threshold_precision(0.2) == 1
    assert calib.threshold_precision(0.225) == 3
    assert calib.threshold_precision(0.22541916370391846) >= 15

    k1 = calib.policy_simplicity_key(0.0, 0.2)
    k2 = calib.policy_simplicity_key(0.0512345, 0.2)
    # k1 is simpler than k2
    assert k1[0] < k2[0]


def test_search_optimal_policy_predeclared_objective():
    # Construct 10 supported queries (8 correct, 2 wrong) and 10 negative queries
    queries = []
    # High confidence correct (sim=0.6, margin=0.15)
    for i in range(4):
        queries.append(synthetic_query(f"c_high_{i}", "hemoglobin", sim=0.6, margin=0.15, top1_correct=True))
    # Medium confidence correct (sim=0.4, margin=0.08)
    for i in range(4):
        queries.append(synthetic_query(f"c_med_{i}", "hemoglobin", sim=0.4, margin=0.08, top1_correct=True))
    # Wrong supported query with high similarity but low margin (sim=0.65, margin=0.05)
    queries.append(synthetic_query("w_high_sim", "hemoglobin", sim=0.65, margin=0.05, top1_correct=False))
    queries.append(synthetic_query("w_other", "hemoglobin", sim=0.45, margin=0.12, top1_correct=False))
    # Negative queries
    for i in range(10):
        # 1 negative query has sim=0.42, margin=0.14
        sim = 0.42 if i == 0 else 0.1
        margin = 0.14 if i == 0 else 0.02
        queries.append(synthetic_query(f"neg_{i}", None, sim=sim, margin=margin))

    cand_sims = [0.0, 0.35, 0.5, 1.0]
    cand_margins = [0.0, 0.10, 0.20, 1.0]

    # Search with max_negative_false_accept_rate=0.05 (allowing <= 0.5 out of 10 -> 0 allowed!)
    # and min_accepted_precision=0.95
    res = calib.search_optimal_policy(queries, cand_sims, cand_margins, max_negative_false_accept_rate=0.05, min_accepted_precision=0.95)

    assert res["found"] is True
    assert res["similarity_threshold"] == 0.5
    assert res["margin_threshold"] == 0.10
    m = res["calibration_metrics"]
    assert m["correct_accept_count"] == 4
    assert m["wrong_accept_count"] == 0
    assert m["negative_false_accept_count"] == 0
    assert m["accepted_precision"] == 1.0
    assert m["supported_correct_coverage"] == pytest.approx(4 / 10)


def test_search_optimal_policy_no_qualifying():
    # All queries are negative with high similarity and margin -> impossible to satisfy prec >= 0.95
    queries = [
        synthetic_query("n1", None, sim=0.8, margin=0.5),
        synthetic_query("n2", None, sim=0.8, margin=0.5),
    ]
    res = calib.search_optimal_policy(queries, [0.0, 0.5], [0.0, 0.3], max_negative_false_accept_rate=0.05, min_accepted_precision=0.95)
    assert res["found"] is False
    assert res["similarity_threshold"] is None
    assert res["margin_threshold"] is None
    assert res["calibration_metrics"] is None


def test_search_optimal_policy_tie_breaking():
    # Setup two policies with equal coverage, but policy A has lower negative_false_accept_rate
    # Candidate thresholds: policy 1: (s=0.2, m=0.1), policy 2: (s=0.3, m=0.1)
    queries = [
        synthetic_query("s1", "hemoglobin", sim=0.5, margin=0.2, top1_correct=True),
        synthetic_query("n1", None, sim=0.25, margin=0.15),  # accepted by s=0.2, rejected by s=0.3
    ]
    # For s=0.2, m=0.1: corr=1, neg_fa=1 -> prec=0.5, neg_rate=1.0
    # For s=0.3, m=0.1: corr=1, neg_fa=0 -> prec=1.0, neg_rate=0.0
    res = calib.search_optimal_policy(queries, [0.2, 0.3], [0.1], max_negative_false_accept_rate=1.0, min_accepted_precision=0.5)
    assert res["found"] is True
    assert res["similarity_threshold"] == 0.3


def test_search_optimal_policy_simplicity_tie_breaking():
    # Setup two policies with identical metrics, but one has simpler threshold representation
    queries = [
        synthetic_query("s1", "hemoglobin", sim=0.5, margin=0.2, top1_correct=True),
    ]
    # s=0.0 is simpler than s=0.12345678
    res = calib.search_optimal_policy(queries, [0.12345678, 0.0], [0.1], max_negative_false_accept_rate=1.0, min_accepted_precision=0.5)
    assert res["found"] is True
    assert res["similarity_threshold"] == 0.0


def test_held_out_firewall_and_calibration_split_isolation():
    # Verify that held_out queries do NOT affect candidate generation or calibration search
    raw_cal = [
        synthetic_query("c1", "hemoglobin", sim=0.5, margin=0.2, top1_correct=True, split="calibration"),
        synthetic_query("n1", None, sim=0.1, margin=0.05, split="calibration"),
    ]
    raw_held = [
        synthetic_query("h1", "hemoglobin", sim=0.99, margin=0.99, top1_correct=True, split="held_out"),
        synthetic_query("h2", None, sim=0.98, margin=0.98, split="held_out"),
    ]

    # Only calibration queries should be passed to calibration routines
    sims, margins = calib.generate_threshold_candidates(raw_cal)
    assert 0.99 not in sims
    assert 0.99 not in margins
    assert 0.98 not in sims
    assert 0.98 not in margins

    res = calib.search_optimal_policy(raw_cal, sims, margins, max_negative_false_accept_rate=1.0, min_accepted_precision=0.5)
    assert res["found"] is True
    assert res["similarity_threshold"] != 0.99


def test_annotate_queries_with_decision():
    queries = [
        synthetic_query("s1", "hemoglobin", sim=0.5, margin=0.2, top1_correct=True),
        synthetic_query("n1", None, sim=0.1, margin=0.05),
    ]
    annotated = calib.annotate_queries_with_decision(queries, similarity_threshold=0.4, margin_threshold=0.15)
    assert len(annotated) == 2
    assert annotated[0]["decision"] == "accept"
    assert annotated[0]["outcome"] == "correct_accept"
    assert annotated[1]["decision"] == "abstain"
    assert annotated[1]["outcome"] == "correct_abstain"


def test_build_calibration_report_synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(calib, "git_revision", lambda: "fake-git-rev")

    # Generate 96 synthetic calibration queries: 56 supported, 40 negative
    rows = []
    for i in range(56):
        doc = DOCUMENT_IDS[i % len(DOCUMENT_IDS)]
        sim = 0.35 + (i / 100.0)
        margin = 0.10 + (i / 200.0)
        rows.append(synthetic_query(f"cal_s_{i}", doc, sim=sim, margin=margin, top1_correct=True))
    for i in range(40):
        rows.append(synthetic_query(f"cal_n_{i}", None, sim=0.15, margin=0.02))

    store = SimpleNamespace(
        kb_directory=KNOWLEDGE_BASE_DIR,
        model_name="fake-model",
        model_revision=None,
        dimension=384,
        collection_name="fake-collection",
    )

    report = calib.build_calibration_report(store=store, rows=rows)

    assert report["calibration"]["name"] == "semantic_retrieval_calibration"
    assert report["calibration"]["dataset_sha256"] == dataset_sha256()
    assert report["calibration"]["kb_sha256"] == knowledge_base_sha256()
    assert report["calibration"]["policy_type"] == "similarity_and_margin"
    assert report["calibration"]["calibration_query_count"] == 96
    assert report["calibration"]["calibration_supported_count"] == 56
    assert report["calibration"]["calibration_negative_count"] == 40

    assert len(report["calibration_queries"]) == 96
    assert "score_distributions" in report
    assert "selected_policy" in report

    out_file = tmp_path / "calibration_report.json"
    calib.write_calibration_report(report, out_file)
    assert out_file.exists()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["calibration"]["name"] == "semantic_retrieval_calibration"


def test_production_benchmark_hashes_and_no_leakage():
    dataset = load_dataset()
    assert len(dataset.queries) == 144
    cal = [q for q in dataset.queries if q.split == "calibration"]
    held = [q for q in dataset.queries if q.split == "held_out"]
    assert len(cal) == 96
    assert len(held) == 48

    groups = {split: {q.group_id for q in dataset.queries if q.split == split} for split in ("calibration", "held_out")}
    assert not (groups["calibration"] & groups["held_out"])

    # Verified frozen dataset and KB hashes
    assert dataset_sha256() == "e481c484da856d1488b2bd728cb4c0732b577d4bea23c3a52156271c2f5e890b"
    assert knowledge_base_sha256() == "f704ff3d180124a90638695413f594531e7609e0a0902257dcd36b10a7c4bcc4"


def test_cli_runner(monkeypatch, capsys):
    mock_report = {
        "selected_policy": {
            "found": True,
            "similarity_threshold": 0.0,
            "margin_threshold": 0.2254,
            "calibration_metrics": {
                "accepted_count": 7,
                "correct_accept_count": 7,
                "wrong_accept_count": 0,
                "negative_false_accept_count": 0,
                "supported_total": 56,
                "negative_total": 40,
                "total": 96,
                "accepted_precision": 1.0,
                "supported_correct_coverage": 0.125,
                "negative_false_accept_rate": 0.0,
                "wrong_supported_accept_rate": 0.0,
                "overall_abstention_rate": 0.9271,
            },
        }
    }
    monkeypatch.setattr(calib, "build_calibration_report", lambda: mock_report)
    monkeypatch.setattr(calib, "write_calibration_report", lambda report: None)

    exit_code = calib.main()
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "similarity >= 0.0" in out
    assert "margin >= 0.2254" in out
