"""Evaluate the public keyword-first hybrid API on 68 keyword + 48 held-out cases."""

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from agents.embedding_service import EMBEDDING_TEXT_VERSION, EmbeddingService
from agents import evaluate_keyword_retrieval as keyword_evaluation
from agents import evaluate_semantic_retrieval as semantic_evaluation
from agents.hybrid_retriever import HybridRetriever
from agents.knowledge_base import KnowledgeBaseLoader
from agents.knowledge_base_fingerprint import BACKEND_DIR, DOCUMENT_IDS, knowledge_base_sha256
from agents.keyword_retriever import KeywordRetriever
from agents.semantic_retriever import FROZEN_POLICY, SemanticRetriever
from agents.vector_store import ChromaVectorStore


REPORT_PATH = BACKEND_DIR / "evaluation_results" / "hybrid_evaluation.json"
OUTCOMES = ("correct_keyword_accept", "correct_semantic_accept", "wrong_keyword_accept",
            "wrong_semantic_accept", "correct_abstain", "missed_supported", "error")
# Frozen fixture provenance from the preceding keyword/semantic evaluations.
FROZEN_HASHES = {
    "keyword_dataset_sha256": "ee70d2ffe910cf58010aa5a8f623c1ac00ab09abdc149fc30f16247966a5420a",
    "semantic_dataset_sha256": "cf799511635cb44add34e8710933142a45a1c717a8f712551f7f42809b1dd0e1",
    "kb_sha256": "f704ff3d180124a90638695413f594531e7609e0a0902257dcd36b10a7c4bcc4",
}


def input_hashes(keyword_path, semantic_path, kb_directory) -> dict:
    return {"keyword_dataset_sha256": keyword_evaluation.dataset_sha256(keyword_path),
            "semantic_dataset_sha256": semantic_evaluation.dataset_sha256(semantic_path),
            "kb_sha256": knowledge_base_sha256(kb_directory)}


def load_primary_cases(keyword_path=keyword_evaluation.DATASET_PATH,
                       semantic_path=semantic_evaluation.DATASET_PATH) -> list[dict]:
    """Primary operational hybrid benchmark: 54 keyword supported + 48 semantic held-out = 102 cases.

    Excludes the 14 keyword-contract negative cases because their labels test exact keyword
    rejection rather than universal semantic irrelevance.
    """
    keyword = keyword_evaluation.load_dataset(keyword_path)
    semantic = semantic_evaluation.load_dataset(semantic_path)
    kw_supported = [q for q in keyword.queries if q.expected_document_id is not None]
    if len(keyword.queries) != 68 or len(kw_supported) != 54:
        raise ValueError(f"Invalid primary composition for keyword: expected 68 total with 54 supported; got {len(keyword.queries)}/{len(kw_supported)}")
    sem_heldout = [q for q in semantic.queries if q.split == "held_out"]
    sem_supported = [q for q in sem_heldout if q.expected_document_id is not None]
    if len(sem_heldout) != 48 or len(sem_supported) != 28:
        raise ValueError(f"Invalid primary composition for semantic held-out: expected 48 total with 28 supported; got {len(sem_heldout)}/{len(sem_supported)}")
    cases = [{**q.model_dump(), "fixture": "keyword", "split": "keyword_contract", "group_id": None}
             for q in kw_supported]
    cases += [{**q.model_dump(), "fixture": "semantic_heldout"}
              for q in sem_heldout]
    if len(cases) != 102 or len({(q['fixture'], q['id']) for q in cases}) != 102:
        raise ValueError(f"Invalid primary composition: expected 102 distinct fixture-qualified cases; got {len(cases)}")
    return cases


def load_keyword_contract_cases(keyword_path=keyword_evaluation.DATASET_PATH) -> list[dict]:
    """All 68 keyword fixture cases for diagnostic evaluation under HybridRetriever."""
    keyword = keyword_evaluation.load_dataset(keyword_path)
    cases = [{**q.model_dump(), "fixture": "keyword", "split": "keyword_contract", "group_id": None}
             for q in keyword.queries]
    if len(cases) != 68 or len({q['id'] for q in cases}) != 68:
        raise ValueError(f"Keyword contract diagnostic requires exactly 68 distinct cases; got {len(cases)}")
    return cases


def load_all_cases(keyword_path=keyword_evaluation.DATASET_PATH,
                   semantic_path=semantic_evaluation.DATASET_PATH) -> list[dict]:
    """All 116 unique evaluation cases across keyword (68) and semantic held-out (48)."""
    keyword = keyword_evaluation.load_dataset(keyword_path)
    semantic = semantic_evaluation.load_dataset(semantic_path)
    cases = [{**q.model_dump(), "fixture": "keyword", "split": "keyword_contract", "group_id": None}
             for q in keyword.queries]
    cases += [{**q.model_dump(), "fixture": "semantic_heldout"}
              for q in semantic.queries if q.split == "held_out"]
    if len(cases) != 116 or len({(q['fixture'], q['id']) for q in cases}) != 116:
        raise ValueError(f"All evaluation cases require exactly 116 distinct cases; got {len(cases)}")
    return cases


def classify_outcome(expected_document_id, found: bool, method: str, document_id) -> str:
    if found:
        if method not in {"keyword", "semantic"} or document_id is None:
            raise ValueError("Inconsistent accepted hybrid result")
        correct = expected_document_id is not None and document_id == expected_document_id
        return f"{'correct' if correct else 'wrong'}_{method}_accept"
    if method != "none" or document_id is not None:
        raise ValueError("Inconsistent abstaining hybrid result")
    return "correct_abstain" if expected_document_id is None else "missed_supported"


class RoutingObserver:
    """Observe component calls without deciding how HybridRetriever routes them."""

    def __init__(self, keyword, semantic):
        self._keyword, self._semantic = keyword, semantic
        self.reset()

    def reset(self):
        self.state = {"keyword_attempted": False, "keyword_hit": None,
                      "semantic_attempted": False, "semantic_accepted": None}

    def keyword(self, query):
        self.state["keyword_attempted"] = True
        result = self._keyword.retrieve(query)
        self.state["keyword_hit"] = result is not None
        return result

    def semantic(self, query):
        self.state["semantic_attempted"] = True
        result = self._semantic.retrieve(query)
        self.state["semantic_accepted"] = result.accepted
        return result


def _decision_details(decision):
    if decision is None:
        return None
    top, second = decision.top_candidate, decision.runner_up
    return {"accepted": decision.accepted,
            "top1_document_id": top.document_id if top else None,
            "top1_similarity": top.similarity if top else None,
            "runner_up_document_id": second.document_id if second else None,
            "runner_up_similarity": second.similarity if second else None,
            "similarity_margin": decision.similarity_margin}


def evaluate_cases(cases: list[dict], retriever: HybridRetriever, observer=None) -> list[dict]:
    rows = []
    for case in cases:
        if observer is not None:
            observer.reset()
        row = {**case, "found": None, "method": None, "returned_document_id": None,
               "correct": False, "outcome": "error", "error": None, "semantic_decision": None}
        try:
            result = retriever.retrieve(case["query"])
            identifier = result.document.id if result.document is not None else None
            outcome = classify_outcome(case["expected_document_id"], result.found, result.method, identifier)
            details = _decision_details(result.semantic_decision)
            row.update(found=result.found, method=result.method, returned_document_id=identifier,
                       outcome=outcome, correct=outcome in {"correct_keyword_accept", "correct_semantic_accept", "correct_abstain"},
                       semantic_decision=details)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        if observer is not None:
            row["routing"] = dict(observer.state)
            row["routing_basis"] = "observed component calls"
        elif row["error"] is None:
            # Test injection can infer successful routing from the public result.
            row["routing"] = {"keyword_attempted": True, "keyword_hit": row["method"] == "keyword",
                              "semantic_attempted": row["method"] != "keyword",
                              "semantic_accepted": result.semantic_decision.accepted if result.semantic_decision else None}
            row["routing_basis"] = "inferred from public result"
        else:
            row["routing"] = dict.fromkeys(("keyword_attempted", "keyword_hit", "semantic_attempted", "semantic_accepted"))
            row["routing_basis"] = "unknown after unobserved error"
        rows.append(row)
    return rows


def _rate(count, total):
    return count / total if total else None


def routing_metrics(rows) -> dict:
    count = lambda key, value: sum(row["routing"][key] is value for row in rows)
    hits, misses = count("keyword_hit", True), count("keyword_hit", False)
    fallbacks = count("semantic_attempted", True)
    return {"keyword_hit_count": hits, "keyword_miss_count": misses,
            "semantic_fallback_count": fallbacks, "semantic_accept_count": count("semantic_accepted", True),
            "semantic_abstain_count": count("semantic_accepted", False),
            "keyword_attempt_count": count("keyword_attempted", True),
            "error_count": sum(row["error"] is not None for row in rows),
            "unknown_routing_count": sum(row["routing_basis"] == "unknown after unobserved error" for row in rows),
            "keyword_hit_semantic_call_count": sum(row["routing"]["keyword_hit"] is True and
                                                     row["routing"]["semantic_attempted"] is True for row in rows),
            "fallback_matches_keyword_misses": fallbacks == misses}


def calculate_metrics(rows: list[dict]) -> dict:
    counts = Counter(row["outcome"] for row in rows)
    total = len(rows)
    supported = sum(row["expected_document_id"] is not None for row in rows)
    negative = total - supported
    correct_accepts = counts["correct_keyword_accept"] + counts["correct_semantic_accept"]
    accepted = sum(row["found"] is True for row in rows)
    negative_false_accepts = sum(row["expected_document_id"] is None and row["found"] is True for row in rows)
    correct = correct_accepts + counts["correct_abstain"]
    result = {
        "total_queries": total, "supported_queries": supported, "negative_queries": negative,
        "correct_keyword_accepts": counts["correct_keyword_accept"], "correct_semantic_accepts": counts["correct_semantic_accept"],
        "wrong_keyword_accepts": counts["wrong_keyword_accept"], "wrong_semantic_accepts": counts["wrong_semantic_accept"],
        "correct_abstentions": counts["correct_abstain"], "missed_supported": counts["missed_supported"],
        "error_count": counts["error"], "total_correct_decisions": correct,
        "overall_decision_accuracy": _rate(correct, total), "accepted_count": accepted,
        "correct_accepted_count": correct_accepts, "accepted_precision": _rate(correct_accepts, accepted),
        "wrong_accept_count": counts["wrong_keyword_accept"] + counts["wrong_semantic_accept"],
        "supported_correct_coverage": _rate(correct_accepts, supported),
        "supported_abstention_rate": _rate(counts["missed_supported"], supported),
        "negative_false_accept_count": negative_false_accepts,
        "negative_false_accept_rate": _rate(negative_false_accepts, negative),
        "negative_abstention_rate": _rate(counts["correct_abstain"], negative),
        "abstention_count": sum(row["found"] is False for row in rows),
        "resolved_by_keyword_percentage": 100 * counts["correct_keyword_accept"] / correct_accepts if correct_accepts else None,
        "resolved_by_semantic_percentage": 100 * counts["correct_semantic_accept"] / correct_accepts if correct_accepts else None,
        "semantic_fallbacks_attempted": routing_metrics(rows)["semantic_fallback_count"],
    }
    for method in ("keyword", "semantic", "none"):
        count = sum(row["method"] == method for row in rows)
        result[f"{method}_method_count"] = count
        result[f"{method}_method_rate"] = _rate(count, total)
    return result


def analyze_rows(rows):
    keyword = [r for r in rows if r["fixture"] == "keyword"]
    keyword_supported = [r for r in keyword if r["expected_document_id"] is not None]
    keyword_negatives = [r for r in keyword if r["expected_document_id"] is None]
    semantic = [r for r in rows if r["fixture"] == "semantic_heldout"]
    primary_rows = [r for r in rows if not (r["fixture"] == "keyword" and r["expected_document_id"] is None)]

    primary_metrics = calculate_metrics(primary_rows)
    primary_routing = routing_metrics(primary_rows)

    diagnostic_cases = []
    for r in keyword_negatives:
        if r["error"] is not None:
            c_outcome = "input_validation_error"
        elif r["method"] == "semantic" and r["found"] is True:
            c_outcome = "semantic_fallback_accepted"
        elif r["found"] is False:
            c_outcome = "semantic_fallback_abstained"
        elif r["method"] == "keyword" and r["found"] is True:
            c_outcome = "keyword_accepted"
        else:
            c_outcome = "unknown"
        diagnostic_cases.append({
            "id": r["id"],
            "query": r["query"],
            "category": r["category"],
            "expected_document_id": r["expected_document_id"],
            "contract_outcome": c_outcome,
            "keyword_rejected": r["routing"]["keyword_hit"] is False or r["error"] is not None,
            "found": r["found"],
            "method": r["method"],
            "returned_document_id": r["returned_document_id"],
            "semantic_decision": r.get("semantic_decision"),
            "error": r.get("error"),
        })

    keyword_contract_diag = {
        "description": (
            "Diagnostic evaluation of all 68 keyword fixture cases under HybridRetriever. "
            "Separates deterministic exact canonical/alias behavior from hybrid semantic retrieval. "
            "The 14 keyword-contract negatives test exact keyword rejection rather than universal semantic irrelevance."
        ),
        "counts": {
            "total_keyword_cases": len(keyword),
            "supported_cases": len(keyword_supported),
            "contract_negative_cases": len(keyword_negatives),
            "exact_keyword_hits": sum(r["routing"]["keyword_hit"] is True for r in keyword),
            "semantic_fallbacks": sum(r["routing"]["semantic_attempted"] is True for r in keyword),
            "semantic_accepts": sum(r["method"] == "semantic" and r["found"] is True for r in keyword),
            "abstentions": sum(r["found"] is False for r in keyword),
            "invalid_input_errors": sum(r["error"] is not None for r in keyword),
            "keyword_correctly_rejected": sum(
                r["routing"]["keyword_hit"] is False or r["error"] is not None
                for r in keyword_negatives
            ),
        },
        "cases": diagnostic_cases,
        "keyword_subset_metrics": calculate_metrics(keyword),
    }

    primary_evaluation = {
        "description": (
            "Primary operational hybrid retrieval benchmark comprising 54 keyword-supported queries "
            "and 48 independent semantic held-out queries (82 supported, 20 negative; total 102). "
            "Excludes keyword-contract negative cases because their labels define deterministic keyword "
            "behavior rather than universal semantic relevance."
        ),
        "counts": {
            "total": len(primary_rows),
            "supported": sum(r["expected_document_id"] is not None for r in primary_rows),
            "negative": sum(r["expected_document_id"] is None for r in primary_rows),
            "keyword_supported": len(keyword_supported),
            "semantic_heldout_supported": sum(r["expected_document_id"] is not None for r in semantic),
            "semantic_heldout_negative": sum(r["expected_document_id"] is None for r in semantic),
        },
        "metrics": primary_metrics,
        "routing": primary_routing,
    }

    negatives = {
        "keyword_negative": keyword_negatives,
        **{category: [r for r in semantic if r["category"] == category]
           for category in semantic_evaluation.NEGATIVE_CATEGORIES},
    }

    return {
        "primary_independent_evaluation": primary_evaluation,
        "primary_metrics": primary_metrics,
        "routing_metrics": primary_routing,
        "keyword_subset": calculate_metrics(keyword),
        "keyword_supported_subset": calculate_metrics(keyword_supported),
        "keyword_contract_diagnostic": keyword_contract_diag,
        "semantic_heldout_subset": calculate_metrics(semantic),
        "supported_category_metrics": {category: calculate_metrics([r for r in semantic if r["category"] == category])
                                       for category in semantic_evaluation.SUPPORTED_CATEGORIES},
        "semantic_category_metrics": {category: calculate_metrics([r for r in semantic if r["category"] == category])
                                      for category in semantic_evaluation.SUPPORTED_CATEGORIES},
        "per_test_metrics": {identifier: calculate_metrics([r for r in semantic if r["expected_document_id"] == identifier])
                             for identifier in DOCUMENT_IDS},
        "negative_category_metrics": {category: calculate_metrics(selected) for category, selected in negatives.items()},
    }


def build_report(*, retriever, store, observer=None,
                 keyword_path=keyword_evaluation.DATASET_PATH, semantic_path=semantic_evaluation.DATASET_PATH,
                 source_directory=None) -> dict:
    hashes = input_hashes(keyword_path, semantic_path, store.kb_directory)
    if hashes != FROZEN_HASHES:
        raise ValueError("Frozen fixture or KB fingerprints changed; evaluation aborted")
    if asdict(FROZEN_POLICY) != {"similarity_threshold": 0.0, "margin_threshold": 0.22541916370391846}:
        raise ValueError("Frozen semantic policy changed")
    all_cases = load_all_cases(keyword_path, semantic_path)
    all_rows = evaluate_cases(all_cases, retriever, observer)
    if input_hashes(keyword_path, semantic_path, store.kb_directory) != hashes:
        raise ValueError("Fixture or KB changed during evaluation")
    primary_rows = [r for r in all_rows if not (r["fixture"] == "keyword" and r["expected_document_id"] is None)]
    analyzed = analyze_rows(all_rows)
    return {"evaluation": {
        "name": "keyword_first_hybrid_primary",
        "protocol_version": "Step 14.2 (102-case primary + keyword contract diagnostic)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **hashes, "git_revision": keyword_evaluation.git_revision(), "python_version": platform.python_version(),
        "embedding_model": store.model_name, "model_revision": store.model_revision, "embedding_dimension": store.dimension,
        "embedding_text_version": EMBEDDING_TEXT_VERSION, "normalized_embeddings": True,
        "chroma_collection_name": store.collection_name, "distance_metric": "cosine",
        "source_index_directory": str(source_directory) if source_directory is not None else None,
        "index_access": "temporary byte copy of existing index; no build or rebuild",
        "error_policy": "Errors remain incorrect in denominators, not abstentions",
        "label_caveat": "Keyword negative labels describe the exact-match contract, not universal semantic unsupportedness. Evaluated separately in keyword_contract_diagnostic.",
    }, "frozen_policy": asdict(FROZEN_POLICY),
        "primary_dataset": {"total": 102, "supported": 82, "negative": 20,
                            "keyword_supported": 54,
                            "semantic_heldout": {"total": 48, "supported": 28, "negative": 20},
                            "keyword_negatives_in_primary": False,
                            "semantic_calibration_included": False},
        **analyzed, "queries": primary_rows, "all_queries": all_rows}


def write_report(report, path=REPORT_PATH):
    serialized = json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def index_file_hashes(directory) -> dict:
    directory = Path(directory)
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _run_snapshot(snapshot_directory, source_directory) -> int:
    store = ChromaVectorStore(snapshot_directory)
    encoder = EmbeddingService(model_name=store.model_name, revision=store.model_revision or "", local_files_only=True)
    observer = RoutingObserver(KeywordRetriever(KnowledgeBaseLoader(store.kb_directory)),
                               SemanticRetriever(encoder, store, KnowledgeBaseLoader(store.kb_directory)))
    retriever = HybridRetriever(SimpleNamespace(retrieve=observer.keyword), SimpleNamespace(retrieve=observer.semantic))
    report = build_report(retriever=retriever, store=store, observer=observer, source_directory=source_directory)
    write_report(report)
    metrics = report["primary_metrics"]
    print(f"Hybrid primary: {metrics['total_correct_decisions']}/{metrics['total_queries']} correct; "
          f"accuracy={metrics['overall_decision_accuracy']:.4f}; errors={metrics['error_count']}")
    print(f"Report: {REPORT_PATH}")
    return int(metrics["error_count"] > 0)


def main() -> int:
    try:
        source = ChromaVectorStore()  # Configuration only: never open the source client.
        if input_hashes(keyword_evaluation.DATASET_PATH, semantic_evaluation.DATASET_PATH, source.kb_directory) != FROZEN_HASHES:
            raise ValueError("Frozen fixture or KB fingerprints changed before real evaluation")
        if not (source.directory / "chroma.sqlite3").is_file():
            raise FileNotFoundError("Existing development index is required; this evaluator never builds one")
        before = index_file_hashes(source.directory)
        # Chroma has no read-only PersistentClient option. Isolate its internal
        # housekeeping in a copy. A child process releases Windows file handles
        # before TemporaryDirectory cleanup; all public hybrid logic is unchanged.
        with tempfile.TemporaryDirectory(prefix="lablens-hybrid-evaluation-") as temporary:
            snapshot = Path(temporary) / "chroma"
            shutil.copytree(source.directory, snapshot)
            if index_file_hashes(snapshot) != before or index_file_hashes(source.directory) != before:
                raise ValueError("Index changed while creating evaluation snapshot")
            code = ("from agents.evaluate_hybrid_retrieval import _run_snapshot; import sys; "
                    "raise SystemExit(_run_snapshot(sys.argv[1], sys.argv[2]))")
            result = subprocess.run([sys.executable, "-B", "-c", code, str(snapshot), str(source.directory)], cwd=BACKEND_DIR)
        if index_file_hashes(source.directory) != before:
            raise ValueError("Development index changed during evaluation")
        print("Development index hashes unchanged; temporary snapshot removed.")
        return result.returncode
    except Exception as exc:
        print(f"Hybrid evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
