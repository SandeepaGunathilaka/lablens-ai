"""Evaluate frozen calibration-selected semantic acceptance policy on held-out split only."""

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform

from agents.embedding_service import EMBEDDING_TEXT_VERSION, EmbeddingService
from agents.evaluate_keyword_retrieval import git_revision
from agents.evaluate_semantic_retrieval import (
    DATASET_PATH,
    NEGATIVE_CATEGORIES,
    SUPPORTED_CATEGORIES,
    SemanticDataset,
    _top_scores,
    dataset_sha256,
    load_dataset,
)
from agents.knowledge_base import KnowledgeBaseLoader
from agents.knowledge_base_fingerprint import BACKEND_DIR, DOCUMENT_IDS, knowledge_base_sha256
from agents.semantic_retriever import SemanticRetriever
from agents.vector_store import ChromaVectorStore


FROZEN_SIMILARITY_THRESHOLD = 0.0
FROZEN_MARGIN_THRESHOLD = 0.22541916370391846
HELDOUT_REPORT_PATH = BACKEND_DIR / "evaluation_results" / "semantic_acceptance_heldout.json"
CALIBRATION_REPORT_PATH = BACKEND_DIR / "evaluation_results" / "semantic_calibration.json"


@dataclass(frozen=True)
class FrozenAcceptancePolicy:
    """Immutable acceptance policy selected on calibration data."""
    similarity_threshold: float = FROZEN_SIMILARITY_THRESHOLD
    margin_threshold: float = FROZEN_MARGIN_THRESHOLD

    def accepts(self, similarity: float | None, margin: float | None) -> bool:
        if similarity is None or margin is None:
            return False
        if not math.isfinite(similarity) or not math.isfinite(margin):
            return False
        return similarity >= self.similarity_threshold and margin >= self.margin_threshold


def classify_heldout_outcome(
    is_supported: bool,
    top1_correct: bool | None,
    accepted: bool,
) -> str:
    """Classify decision outcome into the required mutually exclusive categories."""
    if is_supported:
        if accepted:
            return "correct_accept" if top1_correct else "wrong_accept"
        else:
            return "abstain_correct_candidate" if top1_correct else "abstain_wrong_candidate"
    else:
        return "false_accept" if accepted else "correct_abstain"


def calculate_heldout_metrics(queries: list[dict]) -> dict:
    """Compute overall held-out acceptance, precision, coverage, and error rates."""
    supported_queries = [q for q in queries if q.get("expected_document_id") is not None]
    negative_queries = [q for q in queries if q.get("expected_document_id") is None]

    heldout_total = len(queries)
    supported_total = len(supported_queries)
    negative_total = len(negative_queries)

    correct_accept_count = sum(1 for q in supported_queries if q.get("outcome") == "correct_accept")
    wrong_accept_count = sum(1 for q in supported_queries if q.get("outcome") == "wrong_accept")
    negative_false_accept_count = sum(1 for q in negative_queries if q.get("outcome") == "false_accept")

    abstain_correct_candidate_count = sum(1 for q in supported_queries if q.get("outcome") == "abstain_correct_candidate")
    abstain_wrong_candidate_count = sum(1 for q in supported_queries if q.get("outcome") == "abstain_wrong_candidate")
    correct_abstain_count = sum(1 for q in negative_queries if q.get("outcome") == "correct_abstain")

    accepted_count = correct_accept_count + wrong_accept_count + negative_false_accept_count
    abstained_count = heldout_total - accepted_count

    accepted_precision = correct_accept_count / accepted_count if accepted_count > 0 else None
    supported_correct_coverage = correct_accept_count / supported_total if supported_total > 0 else None
    negative_false_accept_rate = negative_false_accept_count / negative_total if negative_total > 0 else None
    wrong_supported_accept_rate = wrong_accept_count / supported_total if supported_total > 0 else None
    overall_abstention_rate = abstained_count / heldout_total if heldout_total > 0 else None
    supported_accept_rate = (correct_accept_count + wrong_accept_count) / supported_total if supported_total > 0 else None
    negative_abstention_rate = correct_abstain_count / negative_total if negative_total > 0 else None

    return {
        "heldout_total": heldout_total,
        "supported_total": supported_total,
        "negative_total": negative_total,
        "accepted_count": accepted_count,
        "abstained_count": abstained_count,
        "correct_accept_count": correct_accept_count,
        "wrong_accept_count": wrong_accept_count,
        "negative_false_accept_count": negative_false_accept_count,
        "supported_correct_accept_count": correct_accept_count,
        "abstain_correct_candidate_count": abstain_correct_candidate_count,
        "abstain_wrong_candidate_count": abstain_wrong_candidate_count,
        "correct_abstain_count": correct_abstain_count,
        "accepted_precision": accepted_precision,
        "supported_correct_coverage": supported_correct_coverage,
        "negative_false_accept_rate": negative_false_accept_rate,
        "wrong_supported_accept_rate": wrong_supported_accept_rate,
        "overall_abstention_rate": overall_abstention_rate,
        "supported_accept_rate": supported_accept_rate,
        "negative_abstention_rate": negative_abstention_rate,
    }


def calculate_per_test_metrics(queries: list[dict]) -> dict:
    """Analyze acceptance performance for each individual supported test on held-out data."""
    per_test = {}
    for doc_id in DOCUMENT_IDS:
        test_queries = [q for q in queries if q.get("expected_document_id") == doc_id]
        total = len(test_queries)
        top1_correct = sum(1 for q in test_queries if bool(q.get("top1_correct")))
        accepted = sum(1 for q in test_queries if bool(q.get("accepted")))
        correct_accept = sum(1 for q in test_queries if q.get("outcome") == "correct_accept")
        wrong_accept = sum(1 for q in test_queries if q.get("outcome") == "wrong_accept")
        abstained = total - accepted
        coverage = correct_accept / total if total > 0 else None
        precision = correct_accept / accepted if accepted > 0 else None

        per_test[doc_id] = {
            "heldout_supported_queries": total,
            "top1_correct": top1_correct,
            "accepted": accepted,
            "correct_accept": correct_accept,
            "wrong_accept": wrong_accept,
            "abstained": abstained,
            "supported_correct_coverage": coverage,
            "accepted_precision": precision,
        }
    return per_test


def calculate_category_metrics(queries: list[dict]) -> dict:
    """Analyze acceptance performance across each supported and negative query category."""
    categories = {}
    for cat in (*SUPPORTED_CATEGORIES, *NEGATIVE_CATEGORIES):
        cat_queries = [q for q in queries if q.get("category") == cat]
        total = len(cat_queries)
        accepted = sum(1 for q in cat_queries if bool(q.get("accepted")))
        abstained = total - accepted

        if cat in SUPPORTED_CATEGORIES:
            correct_accept = sum(1 for q in cat_queries if q.get("outcome") == "correct_accept")
            wrong_accept = sum(1 for q in cat_queries if q.get("outcome") == "wrong_accept")
            coverage = correct_accept / total if total > 0 else None
            categories[cat] = {
                "total": total,
                "accepted": accepted,
                "abstained": abstained,
                "correct_accept": correct_accept,
                "wrong_accept": wrong_accept,
                "supported_correct_coverage": coverage,
            }
        else:
            false_accept = sum(1 for q in cat_queries if q.get("outcome") == "false_accept")
            false_accept_rate = false_accept / total if total > 0 else None
            abstention_rate = abstained / total if total > 0 else None
            categories[cat] = {
                "total": total,
                "accepted": accepted,
                "abstained": abstained,
                "false_accept": false_accept,
                "false_accept_rate": false_accept_rate,
                "abstention_rate": abstention_rate,
            }
    return categories


def compare_calibration_vs_heldout(cal_metrics: dict, held_metrics: dict) -> dict:
    """Side-by-side comparison of calibration reference and held-out evaluation."""
    keys = (
        "accepted_precision",
        "supported_correct_coverage",
        "negative_false_accept_rate",
        "wrong_supported_accept_rate",
        "overall_abstention_rate",
    )
    comparison = {}
    for key in keys:
        comparison[key] = {
            "calibration": cal_metrics.get(key),
            "held_out": held_metrics.get(key),
            "difference": (
                held_metrics[key] - cal_metrics[key]
                if (held_metrics.get(key) is not None and cal_metrics.get(key) is not None)
                else None
            ),
        }
    return comparison


def evaluate_heldout_queries(
    dataset: SemanticDataset,
    retriever: SemanticRetriever,
    policy: FrozenAcceptancePolicy,
) -> list[dict]:
    """Execute ranked search for held-out queries only and apply immutable frozen policy."""
    held_queries = [
        q for q in dataset.queries
        if (getattr(q, "split", None) if not isinstance(q, dict) else q.get("split")) == "held_out"
    ]
    if len(held_queries) != 48:
        raise ValueError(f"Expected exactly 48 held-out queries, got {len(held_queries)}")

    rows = []
    for query in held_queries:
        query_text = query.query if hasattr(query, "query") else query["query"]
        expected_doc_id = query.expected_document_id if hasattr(query, "expected_document_id") else query.get("expected_document_id")
        dump = query.model_dump() if hasattr(query, "model_dump") else (copy.deepcopy(query) if isinstance(query, dict) else vars(query))

        candidates, error = [], None
        try:
            results = retriever.search(query_text, n_results=7)
            if results and (len(results) != 7 or {r.document_id for r in results} != set(DOCUMENT_IDS)):
                raise ValueError("Evaluation requires all seven distinct approved candidates")
            if any(r.rank != rank or not math.isfinite(r.distance) or not math.isfinite(r.similarity)
                   for rank, r in enumerate(results, start=1)):
                raise ValueError("Invalid candidate rank or nonfinite score")
            candidates = [{
                "rank": r.rank,
                "document_id": r.document_id,
                "test_name": r.test_name,
                "distance": r.distance,
                "similarity": r.similarity,
            } for r in results]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        scores = _top_scores(candidates)
        top1_doc = scores.get("top1_document_id")
        is_supp = expected_doc_id is not None
        top1_corr = (top1_doc == expected_doc_id) if is_supp else None

        top1_sim = scores.get("top1_similarity")
        margin = scores.get("similarity_margin")
        accepted = policy.accepts(top1_sim, margin)
        outcome = classify_heldout_outcome(is_supp, top1_corr, accepted)

        row = {
            **dump,
            "ranked_candidates": candidates,
            "error": error,
            **scores,
            "top1_correct": top1_corr,
            "accepted": accepted,
            "outcome": outcome,
        }
        rows.append(row)
    return rows


def build_heldout_report(
    dataset_path: Path | str = DATASET_PATH,
    calibration_report_path: Path | str = CALIBRATION_REPORT_PATH,
    *,
    store=None,
    retriever=None,
    rows: list[dict] | None = None,
    policy: FrozenAcceptancePolicy | None = None,
) -> dict:
    """Build held-out acceptance evaluation report with calibration reference comparison."""
    dataset_hash = dataset_sha256(dataset_path)
    dataset = load_dataset(dataset_path)

    store = store if store is not None else ChromaVectorStore()
    kb_hash = knowledge_base_sha256(store.kb_directory)

    # Load frozen policy and calibration metrics from calibration report
    cal_path = Path(calibration_report_path)
    cal_data = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else {}
    cal_hash = hashlib.sha256(cal_path.read_bytes()).hexdigest() if cal_path.exists() else None

    cal_policy = cal_data.get("selected_policy", {})
    cal_metrics = cal_policy.get("calibration_metrics", {})

    if policy is None:
        sim_th = cal_policy.get("similarity_threshold", FROZEN_SIMILARITY_THRESHOLD)
        margin_th = cal_policy.get("margin_threshold", FROZEN_MARGIN_THRESHOLD)
        policy = FrozenAcceptancePolicy(similarity_threshold=sim_th, margin_threshold=margin_th)

    if rows is None:
        if retriever is None:
            encoder = EmbeddingService(
                model_name=store.model_name,
                revision=store.model_revision or "",
                local_files_only=True,
            )
            retriever = SemanticRetriever(encoder, store, KnowledgeBaseLoader(store.kb_directory))
        rows = evaluate_heldout_queries(dataset, retriever, policy)

    if dataset_sha256(dataset_path) != dataset_hash or knowledge_base_sha256(store.kb_directory) != kb_hash:
        raise ValueError("Evaluation inputs changed during execution; rerun with stable files")

    held_metrics = calculate_heldout_metrics(rows)
    per_test_metrics = calculate_per_test_metrics(rows)
    category_metrics = calculate_category_metrics(rows)
    comparison = compare_calibration_vs_heldout(cal_metrics, held_metrics)

    # Generalization interpretation
    prec_retained = (
        held_metrics["accepted_precision"] is not None
        and cal_metrics.get("accepted_precision") is not None
        and held_metrics["accepted_precision"] >= cal_metrics["accepted_precision"]
    )
    cov_diff = (
        held_metrics["supported_correct_coverage"] - cal_metrics.get("supported_correct_coverage", 0)
        if held_metrics["supported_correct_coverage"] is not None and cal_metrics.get("supported_correct_coverage") is not None
        else None
    )
    interpretation = {
        "precision_status": (
            "unavailable" if held_metrics["accepted_precision"] is None
            or cal_metrics.get("accepted_precision") is None
            else ("retained" if prec_retained else "reduced")
        ),
        "coverage_status": (
            "unavailable" if cov_diff is None
            else "consistent" if abs(cov_diff) < 0.05
            else "higher" if cov_diff > 0 else "reduced"
        ),
        "negative_safety_status": (
            "zero_false_accepts" if held_metrics["negative_false_accept_count"] == 0 else "false_accepts_present"
        ),
        "heldout_supported_coverage_pct": round(held_metrics["supported_correct_coverage"] * 100, 2) if held_metrics["supported_correct_coverage"] is not None else None,
        "heldout_accepted_precision_pct": round(held_metrics["accepted_precision"] * 100, 2) if held_metrics["accepted_precision"] is not None else None,
    }

    report = {
        "evaluation": {
            "name": "semantic_acceptance_heldout",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_version": dataset.dataset_version,
            "dataset_sha256": dataset_hash,
            "kb_sha256": kb_hash,
            "calibration_report_sha256": cal_hash,
            "git_revision": git_revision(),
            "policy_source": "semantic_calibration.json",
            "embedding_model": store.model_name,
            "model_revision": store.model_revision,
            "embedding_dimension": store.dimension,
            "normalized_embeddings": True,
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
            "chroma_collection_name": store.collection_name,
            "distance_metric": "cosine",
            "python_version": platform.python_version(),
            "heldout_query_count": len(rows),
            "heldout_supported_count": sum(1 for r in rows if r.get("expected_document_id") is not None),
            "heldout_negative_count": sum(1 for r in rows if r.get("expected_document_id") is None),
        },
        "frozen_policy": {
            "similarity_threshold": policy.similarity_threshold,
            "margin_threshold": policy.margin_threshold,
        },
        "calibration_reference_metrics": cal_metrics,
        "heldout_metrics": held_metrics,
        "per_test_metrics": per_test_metrics,
        "category_metrics": category_metrics,
        "calibration_vs_heldout_comparison": comparison,
        "generalization_summary": interpretation,
        "queries": rows,
    }
    return report


def write_heldout_report(report: dict, path: Path | str = HELDOUT_REPORT_PATH) -> None:
    """Serialize held-out evaluation report with deterministic formatting."""
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def main() -> int:
    try:
        report = build_heldout_report()
        write_heldout_report(report)
    except Exception as exc:
        print(f"Held-out acceptance evaluation failed: {exc}")
        return 1

    policy = report["frozen_policy"]
    m = report["heldout_metrics"]
    print(f"Frozen semantic acceptance policy: similarity >= {policy['similarity_threshold']}, margin >= {policy['margin_threshold']}")
    print(f"Held-out Accepted: {m['accepted_count']}/{m['heldout_total']} (Supported: {m['correct_accept_count'] + m['wrong_accept_count']}/{m['supported_total']}, Negative: {m['negative_false_accept_count']}/{m['negative_total']})")
    prec_str = f"{m['accepted_precision']:.4f}" if m['accepted_precision'] is not None else "None"
    print(f"Precision: {prec_str}; Coverage: {m['supported_correct_coverage']:.4f}; Neg False-Accept Rate: {m['negative_false_accept_rate']:.4f}; Wrong Accept Rate: {m['wrong_supported_accept_rate']:.4f}; Abstention Rate: {m['overall_abstention_rate']:.4f}")
    print(f"Report: {HELDOUT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
