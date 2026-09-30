"""Fixture integrity and independent metric arithmetic; no real encoder or index."""

from collections import Counter
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents import evaluate_semantic_retrieval as evaluation
from agents.knowledge_base import KNOWLEDGE_BASE_DIR, KnowledgeBaseLoader
from agents.keyword_retriever import normalize_test_name
from agents.knowledge_base_fingerprint import DOCUMENT_IDS, knowledge_base_sha256


@pytest.fixture(autouse=True)
def no_real_components(monkeypatch):
    for name in ("EmbeddingService", "ChromaVectorStore", "SemanticRetriever"):
        monkeypatch.setattr(evaluation, name, Mock(side_effect=AssertionError("Real runtime forbidden in unit tests")))


def query(identifier="q1", label="hemoglobin", category="direct_semantic", split="calibration"):
    return {"id": identifier, "query": "independent synthetic wording " + identifier,
            "category": category, "expected_document_id": label, "split": split, "group_id": identifier}


def dataset(rows):
    return {"dataset_version": "1.0", "description": "Synthetic test fixture", "queries": rows}


def save(tmp_path, data):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def ranked(target="hemoglobin", rank=1):
    ids = list(DOCUMENT_IDS)
    ids.remove(target)
    ids.insert(rank - 1, target)
    return [SimpleNamespace(document_id=identifier, test_name=identifier, rank=i,
                            distance=i / 10, similarity=1 - i / 10) for i, identifier in enumerate(ids, 1)]


@pytest.fixture
def production():
    return json.loads(evaluation.DATASET_PATH.read_text(encoding="utf-8"))


def test_production_fixture_structure_and_label_independence(production):
    loaded = evaluation.load_dataset()
    assert len(loaded.queries) == 144
    assert Counter(row.category for row in loaded.queries) == evaluation.CATEGORY_COUNTS
    assert Counter(row.split for row in loaded.queries) == {"calibration":96, "held_out":48}
    assert len({row.query for row in loaded.queries}) == 144
    supported = [row for row in loaded.queries if row.expected_document_id is not None]
    assert Counter(row.expected_document_id for row in supported) == dict.fromkeys(DOCUMENT_IDS, 12)
    names = {normalize_test_name(name) for doc in KnowledgeBaseLoader().load_all() for name in [doc.test_name, *doc.aliases]}
    assert all(normalize_test_name(row.query) not in names for row in supported)
    groups = {split:{row.group_id for row in loaded.queries if row.split == split} for split in evaluation.SPLITS}
    assert not groups["calibration"] & groups["held_out"]
    # The labels are read directly from the fixed fixture, without invoking any retriever.
    assert loaded.queries[0].expected_document_id == "hemoglobin"


def test_query_whitespace_preserved(tmp_path):
    row = query()
    row["query"] = "  Synthetic  query  "
    loaded = evaluation.load_dataset(save(tmp_path, dataset([row])), production=False)
    assert loaded.queries[0].query == row["query"]


@pytest.mark.parametrize("field,value", [
    ("category", "invalid"), ("split", "test"), ("expected_document_id", "unknown"),
    ("expected_document_id", None), ("group_id", " "), ("id", ""), ("query", " \t"),
    ("query", 123), ("category", "unsupported_medical"),
])
def test_invalid_query_fields(tmp_path, field, value):
    row = query()
    row[field] = value
    with pytest.raises(ValueError):
        evaluation.load_dataset(save(tmp_path, dataset([row])), production=False)


@pytest.mark.parametrize("payload", [{}, dataset([]), {"dataset_version":" ", "description":"x", "queries":[query()]},
                                      {"description":"x", "queries":[query()]}, dataset([query(), query()])])
def test_invalid_dataset(tmp_path, payload):
    with pytest.raises(ValueError):
        evaluation.load_dataset(save(tmp_path, payload), production=False)


def test_group_leakage_rejected(tmp_path):
    a, b = query(), query("q2", split="held_out")
    b["group_id"] = a["group_id"]
    with pytest.raises(ValueError, match="leakage"):
        evaluation.load_dataset(save(tmp_path, dataset([a,b])), production=False)


@pytest.mark.parametrize("failure", ["total", "category", "split", "per_test", "category_test", "category_test_split", "negative_split"])
def test_exact_distributions(tmp_path, production, failure):
    rows = production["queries"]
    if failure == "total":
        rows.pop()
    elif failure == "category":
        rows[0]["category"] = "paraphrase"
    elif failure == "split":
        rows[0].update(split="held_out", group_id="unique_changed")
    elif failure == "per_test":
        rows[0]["expected_document_id"] = "wbc"
    elif failure == "category_test":
        rows[0]["category"], rows[15]["category"] = rows[15]["category"], rows[0]["category"]
    elif failure == "category_test_split":
        rows[0].update(split="held_out", group_id="unique_changed_a")
        rows[5].update(split="calibration", group_id="unique_changed_b")
    else:
        medical = next(row for row in rows if row["category"] == "unsupported_medical" and row["split"] == "calibration")
        unrelated = next(row for row in rows if row["category"] == "unrelated" and row["split"] == "held_out")
        medical.update(split="held_out", group_id="unique_changed_c")
        unrelated.update(split="calibration", group_id="unique_changed_d")
    with pytest.raises(ValueError, match="counts|distribution|144"):
        evaluation.load_dataset(save(tmp_path, production))


@pytest.fixture
def evaluated():
    cases = [query(str(i), identifier, evaluation.SUPPORTED_CATEGORIES[i % 4], "held_out" if i % 2 else "calibration")
             for i, identifier in enumerate(DOCUMENT_IDS)]
    cases += [query("negative_med", None, "unsupported_medical"), query("negative_other", None, "unrelated", "held_out")]
    semantic = Mock()
    semantic.search.side_effect = [ranked("hemoglobin",1), ranked("wbc",3), ranked("platelets",4),
                                  RuntimeError("offline failure"), [], ranked("triglycerides",7),
                                  ranked("total_cholesterol",2), ranked(), ranked("wbc")]
    keyword = Mock()
    keyword.retrieve.side_effect = [SimpleNamespace(id="hemoglobin"), None, SimpleNamespace(id="wbc"),
                                   RuntimeError("keyword failure"), None, None, None]
    rows = evaluation.evaluate_queries(evaluation.SemanticDataset.model_validate(dataset(cases)), semantic, keyword)
    return rows, semantic, keyword


def test_requests_seven_for_every_query(evaluated):
    rows, semantic, keyword = evaluated
    assert semantic.search.call_count == 9
    assert all(call.kwargs == {"n_results":7} for call in semantic.search.call_args_list)
    assert keyword.retrieve.call_count == 7
    assert len(rows) == 9
    assert rows[0]["ranked_candidates"][0]["document_id"] == "hemoglobin"


def test_rank_metrics_and_denominators(evaluated):
    rows = evaluated[0]
    assert [row["expected_rank"] for row in rows[:7]] == [1,3,4,None,None,7,2]
    assert [row["reciprocal_rank"] for row in rows[:7]] == [1,1/3,1/4,0,0,1/7,1/2]
    report = evaluation.calculate_metrics(rows)
    result = report["supported_metrics"]
    assert result["supported_query_count"] == 7
    assert result["top1_correct"] == 1
    assert result["top1_accuracy"] == 1/7
    assert result["hit_at_3_count"] == 3
    assert result["hit_at_3"] == result["recall_at_3"] == 3/7
    assert result["mrr"] == pytest.approx((1+1/3+1/4+1/7+1/2)/7)
    assert result["error_count"] == 1
    assert result["macro_supported_test_top1_accuracy"] == 1/7


def test_category_per_test_and_split_metrics(evaluated):
    result = evaluation.calculate_metrics(evaluated[0])
    assert result["per_test_metrics"]["wbc"]["mrr"] == 1/3
    assert result["per_test_metrics"]["hdl"]["top1_accuracy"] == 0
    assert result["category_metrics"]["direct_semantic"]["total"] == 2
    assert result["category_metrics"]["direct_semantic"]["top1_accuracy"] == .5
    assert result["split_metrics"]["calibration"]["total"] == 4
    assert result["split_metrics"]["held_out"]["total"] == 3
    assert result["split_metrics"]["calibration"]["top1_accuracy"] == .25
    assert result["split_metrics"]["held_out"]["mrr"] == pytest.approx((1/3+1/7)/3)


def test_negative_stats_and_margin(evaluated):
    rows = evaluated[0]
    row = rows[-1]
    assert row["top1_document_id"] == "wbc"
    assert row["top1_similarity"] == .9
    assert row["top1_distance"] == .1
    assert row["similarity_margin"] == pytest.approx(.1)
    assert row["similarity_margin"] == pytest.approx(row["top2_distance"] - row["top1_distance"])
    assert "top1_correct" not in row
    result = evaluation.calculate_metrics(rows)["negative_analysis"]["unrelated"]
    assert result["top1_similarity"] == {"count":1,"min":.9,"max":.9,"mean":.9,"median":.9}
    assert "accuracy" not in result
    assert "rejected" not in result


def test_descriptive_statistics():
    assert evaluation.descriptive_statistics([.1,.3,.8]) == {
        "count":3,"min":.1,"max":.8,"mean":pytest.approx(.4),"median":.3}
    assert evaluation.descriptive_statistics([]) == {"count":0,"min":None,"max":None,"mean":None,"median":None}


def test_correct_incorrect_score_groups_exclude_missing_scores(evaluated):
    result = evaluation.calculate_metrics(evaluated[0])
    correct = result["supported_score_analysis"]["correct_top1"]
    incorrect = result["supported_score_analysis"]["incorrect_top1"]
    assert correct["query_count"] == correct["top1_similarity"]["count"] == 1
    assert incorrect["query_count"] == 6
    assert incorrect["top1_similarity"]["count"] == 4
    assert incorrect["error_count"] == 1
    assert result["score_analysis_by_split"]["held_out"]["negative"]["unrelated"]["query_count"] == 1


def test_keyword_comparison(evaluated):
    metrics = evaluation.calculate_metrics(evaluated[0])["keyword_comparison"]
    assert metrics["supported_query_count"] == 7
    assert metrics["correct_document_count"] == 1
    assert metrics["correct_document_rate"] == 1/7
    assert metrics["answered_count"] == 2
    assert metrics["answered_rate"] == 2/7
    assert metrics["error_count"] == 1


def test_negative_error_is_not_success():
    cases = evaluation.SemanticDataset.model_validate(dataset([query("n", None, "unrelated")]))
    semantic = Mock(search=Mock(side_effect=RuntimeError("broken")))
    rows = evaluation.evaluate_queries(cases, semantic, Mock())
    assert rows[0]["error"] == "RuntimeError: broken"
    assert rows[0]["top1_similarity"] is None
    metrics = evaluation.calculate_metrics(rows)
    assert metrics["negative_analysis"]["unrelated"]["error_count"] == 1
    assert metrics["negative_analysis"]["unrelated"]["top1_similarity"]["count"] == 0
    assert metrics["supported_metrics"]["mrr"] is None
    assert metrics["supported_metrics"]["macro_supported_test_top1_accuracy"] is None


@pytest.mark.parametrize("failure", ["partial", "duplicate", "rank", "nonfinite"])
def test_bad_rankings_are_recorded_errors(failure):
    results = ranked()
    if failure == "partial":
        results.pop()
    elif failure == "duplicate":
        results[1].document_id = results[0].document_id
    elif failure == "rank":
        results[0].rank = 2
    else:
        results[0].distance = float("nan")
    rows = evaluation.evaluate_queries(evaluation.SemanticDataset.model_validate(dataset([query()])),
                                       Mock(search=Mock(return_value=results)), Mock(retrieve=Mock(return_value=None)))
    assert rows[0]["error"] is not None
    assert rows[0]["reciprocal_rank"] == 0
    assert rows[0]["ranked_candidates"] == []


def test_dataset_hash_deterministic_and_byte_sensitive(tmp_path):
    path = save(tmp_path, dataset([query()]))
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert evaluation.dataset_sha256(path) == evaluation.dataset_sha256(path) == expected
    path.write_bytes(path.read_bytes()+b"\n")
    assert evaluation.dataset_sha256(path) != expected


def test_complete_report_metadata_serialization_and_no_real_components(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "git_revision", lambda: None)
    store = SimpleNamespace(kb_directory=KNOWLEDGE_BASE_DIR, model_name="fake-model", model_revision="revision1",
                            dimension=384, collection_name="fake_collection")
    report = evaluation.build_report(store=store, retriever=Mock(search=Mock(return_value=ranked())),
                                    keyword_retriever=Mock(retrieve=Mock(return_value=None)))
    assert report["dataset_counts"]["total"] == 144
    assert report["dataset_counts"]["splits"] == {"calibration":{"total":96,"supported":56,"negative":40},
                                                    "held_out":{"total":48,"supported":28,"negative":20}}
    meta = report["evaluation"]
    assert meta["git_revision"] is None
    assert meta["generated_at"].endswith("+00:00")
    assert meta["dataset_sha256"] == evaluation.dataset_sha256()
    assert meta["knowledge_base_sha256"] == knowledge_base_sha256()
    assert meta["embedding_model"] == "fake-model"
    assert meta["model_revision"] == "revision1"
    assert meta["embedding_dimension"] == 384
    assert meta["normalized_embeddings"] is True
    assert meta["embedding_text_version"] == "1"
    assert meta["chroma_collection_name"] == "fake_collection"
    assert meta["distance_metric"] == "cosine"
    assert meta["python_version"]
    path = tmp_path / "output" / "report.json"
    evaluation.write_report(report, path)
    assert json.loads(path.read_text()) == report


def test_changed_inputs_rejected(monkeypatch):
    monkeypatch.setattr(evaluation, "dataset_sha256", Mock(side_effect=["before", "after"]))
    store = SimpleNamespace(kb_directory=KNOWLEDGE_BASE_DIR)
    with pytest.raises(ValueError, match="changed"):
        evaluation.build_report(store=store, retriever=Mock(search=Mock(return_value=ranked())),
                                keyword_retriever=Mock(retrieve=Mock(return_value=None)))


@pytest.mark.parametrize("errors,keyword_errors,exit_code", [(0,0,0),(1,0,1),(0,1,1)])
def test_cli_summary_and_exit_codes(monkeypatch, capsys, errors, keyword_errors, exit_code):
    report = {"supported_metrics":{"top1_correct":1,"total":2,"top1_accuracy":.5,"hit_at_3":1.,"mrr":.75},
              "dataset_counts":{"semantic_error_count":errors},"keyword_comparison":{"error_count":keyword_errors}}
    monkeypatch.setattr(evaluation, "build_report", lambda: report)
    writer = Mock()
    monkeypatch.setattr(evaluation, "write_report", writer)
    assert evaluation.main() == exit_code
    writer.assert_called_once_with(report)
    assert "Top-1: 0.5000" in capsys.readouterr().out


def test_cli_setup_failure(monkeypatch, capsys):
    monkeypatch.setattr(evaluation, "build_report", Mock(side_effect=ValueError("invalid fixture")))
    writer = Mock()
    monkeypatch.setattr(evaluation, "write_report", writer)
    assert evaluation.main() == 1
    assert "invalid fixture" in capsys.readouterr().out
    writer.assert_not_called()
