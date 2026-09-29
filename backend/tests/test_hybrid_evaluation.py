"""Offline hybrid evaluation arithmetic and public-API routing checks."""

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents import evaluate_hybrid_retrieval as evaluation
from agents.hybrid_retriever import HybridRetrievalResult
from agents.knowledge_base import KNOWLEDGE_BASE_DIR
from agents.semantic_retriever import SemanticRetrievalDecision


@pytest.fixture(autouse=True)
def no_real_components(monkeypatch):
    for name in ("ChromaVectorStore", "EmbeddingService", "SemanticRetriever", "HybridRetriever"):
        monkeypatch.setattr(evaluation, name, Mock(side_effect=AssertionError("Real components forbidden")))


def case(identifier, expected, fixture="semantic_heldout", category="description"):
    return {"id":identifier, "query":"synthetic " + identifier, "fixture":fixture,
            "category":category, "split":"held_out" if fixture == "semantic_heldout" else "keyword_contract",
            "group_id":identifier, "expected_document_id":expected}


def result(method, identifier=None):
    document = SimpleNamespace(id=identifier) if identifier is not None else None
    decision = None if method == "keyword" else SemanticRetrievalDecision(
        method == "semantic", document, SimpleNamespace(document_id=identifier or "wbc", similarity=.6),
        SimpleNamespace(document_id="ldl", similarity=.3), .3)
    return HybridRetrievalResult(document, identifier is not None, method, decision)


@pytest.fixture
def evaluated():
    cases = [case("kw_ok", "hemoglobin", "keyword", "canonical"),
             case("kw_wrong", "wbc", "keyword", "alias"),
             case("sem_ok", "platelets"), case("sem_wrong", "hdl"),
             case("kw_negative", None, "keyword", "unsupported"),
             case("sem_negative", None, category="unsupported_medical"),
             case("abstain", None, category="unrelated"), case("miss", "ldl"),
             case("error", None, "keyword", "unsupported")]
    hybrid = Mock(retrieve=Mock(side_effect=[result("keyword","hemoglobin"), result("keyword","platelets"),
        result("semantic","platelets"), result("semantic","ldl"), result("keyword","wbc"),
        result("semantic","hdl"), result("none"), result("none"), RuntimeError("broken")]))
    return evaluation.evaluate_cases(cases, hybrid), hybrid


def test_primary_composition():
    rows = evaluation.load_primary_cases()
    assert len(rows) == 102
    assert sum(r["expected_document_id"] is not None for r in rows) == 82
    assert sum(r["expected_document_id"] is None for r in rows) == 20
    assert sum(r["fixture"] == "keyword" for r in rows) == 54
    assert sum(r["fixture"] == "semantic_heldout" for r in rows) == 48
    assert all(r["split"] == "held_out" for r in rows if r["fixture"] == "semantic_heldout")
    assert all(r["split"] != "calibration" for r in rows)
    assert all(r["expected_document_id"] is not None for r in rows if r["fixture"] == "keyword")
    assert sum(not r["query"].strip() for r in rows) == 0


@pytest.mark.parametrize("fixture", ["keyword", "semantic"])
def test_bad_composition_rejected(monkeypatch, fixture):
    component = evaluation.keyword_evaluation if fixture == "keyword" else evaluation.semantic_evaluation
    loaded = component.load_dataset()
    loaded.queries.pop()
    monkeypatch.setattr(component, "load_dataset", lambda path: loaded)
    with pytest.raises(ValueError, match="composition"):
        evaluation.load_primary_cases()


@pytest.mark.parametrize("expected,found,method,identifier,outcome", [
    ("hemoglobin",True,"keyword","hemoglobin","correct_keyword_accept"),
    ("hemoglobin",True,"semantic","hemoglobin","correct_semantic_accept"),
    ("hemoglobin",True,"keyword","wbc","wrong_keyword_accept"),
    (None,True,"keyword","wbc","wrong_keyword_accept"),
    ("hemoglobin",True,"semantic","wbc","wrong_semantic_accept"),
    (None,True,"semantic","wbc","wrong_semantic_accept"),
    (None,False,"none",None,"correct_abstain"),
    ("hemoglobin",False,"none",None,"missed_supported"),
])
def test_classification(expected, found, method, identifier, outcome):
    assert evaluation.classify_outcome(expected,found,method,identifier) == outcome


@pytest.mark.parametrize("found,method,identifier", [(True,"none","wbc"),(True,"keyword",None),(False,"keyword",None),(False,"none","wbc")])
def test_invalid_public_results_raise(found, method, identifier):
    with pytest.raises(ValueError, match="Inconsistent"):
        evaluation.classify_outcome(None,found,method,identifier)


def test_one_public_call_per_case_and_decision_details(evaluated):
    rows, hybrid = evaluated
    assert hybrid.retrieve.call_count == len(rows)
    assert [call.args[0] for call in hybrid.retrieve.call_args_list] == [r["query"] for r in rows]
    assert all(call[0] == "retrieve" for call in hybrid.mock_calls)
    assert rows[0]["semantic_decision"] is None
    assert rows[2]["semantic_decision"] == {"accepted":True,"top1_document_id":"platelets","top1_similarity":.6,
                                           "runner_up_document_id":"ldl","runner_up_similarity":.3,"similarity_margin":.3}


def test_primary_metric_arithmetic(evaluated):
    metrics = evaluation.calculate_metrics(evaluated[0])
    assert metrics["total_queries"] == 9
    assert metrics["supported_queries"] == 5
    assert metrics["negative_queries"] == 4
    assert metrics["correct_keyword_accepts"] == metrics["correct_semantic_accepts"] == 1
    assert metrics["wrong_keyword_accepts"] == metrics["wrong_semantic_accepts"] == 2
    assert metrics["correct_abstentions"] == metrics["missed_supported"] == 1
    assert metrics["total_correct_decisions"] == 3
    assert metrics["overall_decision_accuracy"] == 3/9
    assert metrics["accepted_count"] == 6
    assert metrics["accepted_precision"] == 2/6
    assert metrics["wrong_accept_count"] == 4
    assert metrics["supported_correct_coverage"] == 2/5
    assert metrics["supported_abstention_rate"] == 1/5
    assert metrics["negative_false_accept_count"] == 2
    assert metrics["negative_false_accept_rate"] == 2/4
    assert metrics["negative_abstention_rate"] == 1/4
    assert metrics["error_count"] == 1


def test_method_and_routing_counts(evaluated):
    rows = evaluated[0]
    metrics = evaluation.calculate_metrics(rows)
    assert metrics["keyword_method_count"] == metrics["semantic_method_count"] == 3
    assert metrics["none_method_count"] == 2
    assert metrics["keyword_method_rate"] == metrics["semantic_method_rate"] == 3/9
    assert metrics["none_method_rate"] == 2/9
    assert metrics["resolved_by_keyword_percentage"] == metrics["resolved_by_semantic_percentage"] == 50
    route = evaluation.routing_metrics(rows)
    assert route["keyword_hit_count"] == 3
    assert route["keyword_miss_count"] == route["semantic_fallback_count"] == 5
    assert route["semantic_accept_count"] == 3
    assert route["semantic_abstain_count"] == 2
    assert route["unknown_routing_count"] == 1
    assert route["fallback_matches_keyword_misses"] is True


def test_subset_category_and_test_metrics(evaluated):
    report = evaluation.analyze_rows(evaluated[0])
    assert report["keyword_subset"]["total_queries"] == 4
    assert report["keyword_subset"]["supported_queries"] == 2
    assert report["keyword_subset"]["overall_decision_accuracy"] == 1/4
    assert report["semantic_heldout_subset"]["total_queries"] == 5
    assert report["semantic_heldout_subset"]["accepted_precision"] == 1/3
    assert report["supported_category_metrics"]["description"]["supported_correct_coverage"] == 1/3
    assert report["per_test_metrics"]["platelets"]["supported_correct_coverage"] == 1
    assert report["per_test_metrics"]["hdl"]["wrong_accept_count"] == 1
    assert report["per_test_metrics"]["ldl"]["missed_supported"] == 1
    assert report["negative_category_metrics"]["keyword_negative"]["negative_false_accept_rate"] == .5
    assert report["negative_category_metrics"]["unsupported_medical"]["negative_false_accept_rate"] == 1
    assert report["negative_category_metrics"]["unrelated"]["correct_abstentions"] == 1


def test_errors_not_abstentions(evaluated):
    row = evaluated[0][-1]
    assert row["outcome"] == "error"
    assert row["error"] == "RuntimeError: broken"
    assert row["found"] is None
    assert row["method"] is None
    assert row["correct"] is False


def test_empty_denominators():
    metrics = evaluation.calculate_metrics([])
    for key in ("accepted_precision","overall_decision_accuracy","supported_correct_coverage",
                "negative_false_accept_rate","negative_abstention_rate","keyword_method_rate",
                "resolved_by_keyword_percentage"):
        assert metrics[key] is None


def test_observer_records_actual_calls_and_failures():
    keyword = Mock(retrieve=Mock(return_value=None))
    semantic = Mock(retrieve=Mock(side_effect=RuntimeError("stale")))
    observer = evaluation.RoutingObserver(keyword, semantic)
    assert observer.keyword("text") is None
    with pytest.raises(RuntimeError):
        observer.semantic("text")
    assert observer.state == {"keyword_attempted":True,"keyword_hit":False,"semantic_attempted":True,"semantic_accepted":None}
    observer.reset()
    assert observer.state["keyword_attempted"] is False
    assert observer.state["semantic_attempted"] is False


def test_observed_keyword_precedence_uses_public_component(monkeypatch):
    # Small fake public API triggers proxies itself; evaluator must not route it.
    keyword = Mock(retrieve=Mock(return_value=SimpleNamespace(id="wbc")))
    semantic = Mock(retrieve=Mock(side_effect=AssertionError("must not run")))
    observer = evaluation.RoutingObserver(keyword, semantic)
    class PublicHybrid:
        def retrieve(self, query):
            return HybridRetrievalResult(observer.keyword(query),True,"keyword",None)
    rows = evaluation.evaluate_cases([case("q","wbc")], PublicHybrid(), observer)
    assert rows[0]["routing"]["keyword_hit"] is True
    semantic.retrieve.assert_not_called()
    assert evaluation.routing_metrics(rows)["keyword_hit_semantic_call_count"] == 0


def test_blank_error_has_no_component_calls():
    observer = evaluation.RoutingObserver(Mock(), Mock())
    hybrid = Mock(retrieve=Mock(side_effect=ValueError("Query must be a nonblank string")))
    rows = evaluation.evaluate_cases([{**case("blank",None),"query":""}], hybrid, observer)
    assert rows[0]["outcome"] == "error"
    route = evaluation.routing_metrics(rows)
    assert route["keyword_attempt_count"] == route["semantic_fallback_count"] == 0
    assert route["error_count"] == 1


def fake_store(tmp_path=None):
    return SimpleNamespace(kb_directory=KNOWLEDGE_BASE_DIR, model_name="fake", model_revision=None,
                           dimension=384, collection_name="fixture", directory=tmp_path)


def test_hashes_match_frozen_provenance():
    hashes = evaluation.input_hashes(evaluation.keyword_evaluation.DATASET_PATH,
                                    evaluation.semantic_evaluation.DATASET_PATH,KNOWLEDGE_BASE_DIR)
    assert hashes == evaluation.FROZEN_HASHES
    assert evaluation.input_hashes(evaluation.keyword_evaluation.DATASET_PATH,
                                   evaluation.semantic_evaluation.DATASET_PATH,KNOWLEDGE_BASE_DIR) == hashes


def test_complete_report_and_serialization(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation.keyword_evaluation,"git_revision",lambda:None)
    hybrid = Mock(retrieve=Mock(return_value=result("none")))
    report = evaluation.build_report(retriever=hybrid,store=fake_store())
    assert hybrid.retrieve.call_count == 116
    assert report["primary_dataset"]["supported"] == 82
    assert report["primary_dataset"]["negative"] == 20
    assert report["primary_dataset"]["total"] == 102
    assert report["primary_dataset"]["semantic_calibration_included"] is False
    assert report["primary_independent_evaluation"]["counts"]["total"] == 102
    assert report["keyword_contract_diagnostic"]["counts"]["total_keyword_cases"] == 68
    assert report["keyword_contract_diagnostic"]["counts"]["contract_negative_cases"] == 14
    assert report["evaluation"]["git_revision"] is None
    assert report["frozen_policy"] == {"similarity_threshold":0.,"margin_threshold":0.22541916370391846}
    assert report["evaluation"]["embedding_dimension"] == 384
    path = tmp_path / "results" / "report.json"
    evaluation.write_report(report,path)
    import json
    assert json.loads(path.read_text()) == report


@pytest.mark.parametrize("when", ["before","during"])
def test_hash_changes_abort(monkeypatch, when):
    values = [{}] if when == "before" else [evaluation.FROZEN_HASHES,{}]
    monkeypatch.setattr(evaluation,"input_hashes",Mock(side_effect=values))
    hybrid = Mock(retrieve=Mock(return_value=result("none")))
    with pytest.raises(ValueError,match="fingerprint|changed"):
        evaluation.build_report(retriever=hybrid,store=fake_store())
    assert hybrid.retrieve.call_count == (0 if when == "before" else 116)


def test_policy_change_aborts(monkeypatch):
    from agents.semantic_retriever import SemanticAcceptancePolicy
    monkeypatch.setattr(evaluation,"FROZEN_POLICY",SemanticAcceptancePolicy(margin_threshold=.5))
    hybrid = Mock()
    with pytest.raises(ValueError,match="policy changed"):
        evaluation.build_report(retriever=hybrid,store=fake_store())
    hybrid.retrieve.assert_not_called()


def test_cli_snapshot_isolated_and_cleaned(tmp_path,monkeypatch):
    source = tmp_path / "development"
    source.mkdir()
    (source / "chroma.sqlite3").write_bytes(b"synthetic database")
    monkeypatch.setattr(evaluation,"ChromaVectorStore",lambda:fake_store(source))
    before = evaluation.index_file_hashes(source)
    snapshots = []
    def run(command, **kwargs):
        snapshot = Path(command[-2])
        snapshots.append(snapshot)
        assert snapshot != source
        assert evaluation.index_file_hashes(snapshot) == before
        (snapshot / "chroma.sqlite3").write_bytes(b"housekeeping only in snapshot")
        return SimpleNamespace(returncode=1)  # Preserve per-case errors in CLI status.
    monkeypatch.setattr(evaluation.subprocess,"run",run)
    assert evaluation.main() == 1
    assert evaluation.index_file_hashes(source) == before
    assert all(not path.exists() for path in snapshots)


def test_missing_index_is_not_created(tmp_path,monkeypatch):
    source = tmp_path / "absent"
    monkeypatch.setattr(evaluation,"ChromaVectorStore",lambda:fake_store(source))
    assert evaluation.main() == 1
    assert not source.exists()

def test_audit_error_and_false_accept_semantics():
    """Verify Step 14.1 audit invariants:
    - negative error is not false accept and not correct abstention
    - negative found=True semantic result is false accept (wrong_semantic_accept)
    - error remains in overall denominator
    - accepted_count counts only found=True
    - accepted_precision excludes errors from numerator and denominator
    - routing counts exclude pre-routing validation errors
    - all outcome categories are mutually exclusive
    """
    cases = [
        case("neg_err", None, "keyword", "unsupported"),
        case("neg_fa_sem", None, "keyword", "unsupported"),
        case("neg_abstain", None, "keyword", "unsupported"),
        case("sup_ok_kw", "hemoglobin", "keyword", "canonical"),
        case("sup_ok_sem", "platelets", "semantic_heldout", "paraphrase"),
        case("sup_wrong_kw", "hemoglobin", "keyword", "alias"),
        case("sup_wrong_sem", "hdl", "semantic_heldout", "description"),
        case("sup_miss", "ldl", "semantic_heldout", "description"),
        case("sup_err", "wbc", "semantic_heldout", "direct_semantic"),
    ]
    hybrid = Mock(retrieve=Mock(side_effect=[
        ValueError("Query must be a nonblank string"),
        result("semantic", "hemoglobin"),
        result("none"),
        result("keyword", "hemoglobin"),
        result("semantic", "platelets"),
        result("keyword", "wbc"),
        result("semantic", "ldl"),
        result("none"),
        RuntimeError("encoder failed"),
    ]))
    rows = evaluation.evaluate_cases(cases, hybrid)
    metrics = evaluation.calculate_metrics(rows)

    # All outcome categories are mutually exclusive and partition the dataset
    outcomes = [r["outcome"] for r in rows]
    assert len(outcomes) == 9
    assert sorted(outcomes) == sorted([
        "error",
        "wrong_semantic_accept",
        "correct_abstain",
        "correct_keyword_accept",
        "correct_semantic_accept",
        "wrong_keyword_accept",
        "wrong_semantic_accept",
        "missed_supported",
        "error",
    ])

    # Negative error is not false accept, and not correct abstention
    neg_err_row = rows[0]
    assert neg_err_row["outcome"] == "error"
    assert neg_err_row["found"] is None
    assert neg_err_row["correct"] is False

    # Negative found=True semantic result is a false accept
    neg_fa_row = rows[1]
    assert neg_fa_row["outcome"] == "wrong_semantic_accept"
    assert neg_fa_row["found"] is True
    assert neg_fa_row["method"] == "semantic"

    # negative_false_accept_count counts only negative queries where found=True
    assert metrics["negative_queries"] == 3
    assert metrics["negative_false_accept_count"] == 1
    assert metrics["negative_false_accept_rate"] == 1 / 3
    assert metrics["negative_abstention_rate"] == 1 / 3

    # Errors remain in overall denominator
    assert metrics["total_queries"] == 9
    assert metrics["error_count"] == 2
    assert metrics["total_correct_decisions"] == 3
    assert metrics["overall_decision_accuracy"] == 3 / 9

    # accepted_count counts only found=True cases
    assert metrics["accepted_count"] == 5
    assert metrics["correct_accepted_count"] == 2
    assert metrics["accepted_precision"] == 2 / 5

    # Routing counts exclude pre-routing validation errors
    obs = evaluation.RoutingObserver(
        Mock(retrieve=Mock(return_value=None)),
        Mock(retrieve=Mock(return_value=SimpleNamespace(accepted=True))),
    )
    class PreRoutingFailingHybrid:
        def retrieve(self, q):
            if not q.strip():
                raise ValueError("blank")
            return HybridRetrievalResult(None, False, "none", None)

    eval_rows = evaluation.evaluate_cases(
        [{"query": "", "expected_document_id": None, "id": "b", "fixture": "keyword"}],
        PreRoutingFailingHybrid(),
        obs,
    )
    route = evaluation.routing_metrics(eval_rows)
    assert route["keyword_attempt_count"] == 0
    assert route["keyword_hit_count"] == 0
    assert route["keyword_miss_count"] == 0
    assert route["semantic_fallback_count"] == 0
    assert route["error_count"] == 1

def test_step_14_2_protocol_requirements():
    """Verify all Step 14.2 protocol specifications:
    - primary total exactly 102 (82 supported, 20 negative)
    - all 54 keyword supported cases included
    - all 48 semantic held-out cases included
    - all 14 keyword-contract negative cases excluded from primary metrics
    - semantic calibration cases excluded
    - keyword-contract negatives remain present in diagnostic section
    - keyword fixture labels remain unchanged
    - unsupported_4 and unsupported_12 remain keyword-contract negatives but do not contribute to primary semantic false-accept metrics
    - unsupported_9 and unsupported_10 remain validation errors in diagnostic analysis
    - primary negative metrics use only 12 unsupported_medical and 8 unrelated
    - primary accepted precision arithmetic
    - primary supported coverage arithmetic
    - primary negative false-accept arithmetic
    """
    primary_cases = evaluation.load_primary_cases()
    assert len(primary_cases) == 102
    assert sum(c["expected_document_id"] is not None for c in primary_cases) == 82
    assert sum(c["expected_document_id"] is None for c in primary_cases) == 20

    # 54 keyword supported + 48 semantic held-out
    assert sum(c["fixture"] == "keyword" for c in primary_cases) == 54
    assert sum(c["fixture"] == "semantic_heldout" for c in primary_cases) == 48
    assert all(c["expected_document_id"] is not None for c in primary_cases if c["fixture"] == "keyword")

    # 14 keyword negatives excluded from primary
    kw_cases = evaluation.load_keyword_contract_cases()
    assert len(kw_cases) == 68
    kw_neg_ids = {c["id"] for c in kw_cases if c["expected_document_id"] is None}
    assert len(kw_neg_ids) == 14
    primary_ids = {c["id"] for c in primary_cases}
    assert kw_neg_ids.isdisjoint(primary_ids)

    # unsupported_4, unsupported_12, unsupported_9, unsupported_10 in keyword negatives
    assert {"unsupported_4", "unsupported_12", "unsupported_9", "unsupported_10"}.issubset(kw_neg_ids)

    # Primary negative categories are exactly 12 unsupported_medical and 8 unrelated
    primary_neg = [c for c in primary_cases if c["expected_document_id"] is None]
    assert len(primary_neg) == 20
    assert sum(c["category"] == "unsupported_medical" for c in primary_neg) == 12
    assert sum(c["category"] == "unrelated" for c in primary_neg) == 8

    # No calibration cases
    assert all(c.get("split") != "calibration" for c in primary_cases)

    # Test report structure with mock
    mock_hybrid = Mock(retrieve=Mock(side_effect=lambda q: result("keyword", "hemoglobin") if q else result("none")))
    report = evaluation.build_report(retriever=mock_hybrid, store=fake_store())

    # Primary evaluation contains 102 cases
    assert report["primary_independent_evaluation"]["counts"]["total"] == 102
    assert report["primary_independent_evaluation"]["counts"]["supported"] == 82
    assert report["primary_independent_evaluation"]["counts"]["negative"] == 20

    # Diagnostic section contains 68 cases, including the 14 negatives
    diag = report["keyword_contract_diagnostic"]
    assert diag["counts"]["total_keyword_cases"] == 68
    assert diag["counts"]["supported_cases"] == 54
    assert diag["counts"]["contract_negative_cases"] == 14
    diag_case_ids = {c["id"] for c in diag["cases"]}
    assert diag_case_ids == kw_neg_ids

    # Negative category metrics has unsupported_medical (12) and unrelated (8)
    assert report["negative_category_metrics"]["unsupported_medical"]["total_queries"] == 12
    assert report["negative_category_metrics"]["unrelated"]["total_queries"] == 8

