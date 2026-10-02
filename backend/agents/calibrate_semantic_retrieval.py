"""Conjunctive acceptance-policy calibration using calibration data only."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
from typing import Annotated

from agents.embedding_service import EMBEDDING_TEXT_VERSION, EmbeddingService
from agents.evaluate_keyword_retrieval import git_revision
from agents.evaluate_semantic_retrieval import (
    DATASET_PATH,
    NEGATIVE_CATEGORIES,
    SUPPORTED_CATEGORIES,
    SemanticDataset,
    SemanticQuery,
    _top_scores,
    dataset_sha256,
    load_dataset,
    score_analysis,
)
from agents.knowledge_base import KnowledgeBaseLoader
from agents.knowledge_base_fingerprint import BACKEND_DIR, DOCUMENT_IDS, knowledge_base_sha256
from agents.semantic_retriever import SemanticRetriever
from agents.vector_store import ChromaVectorStore


CALIBRATION_REPORT_PATH = BACKEND_DIR / "evaluation_results" / "semantic_calibration.json"
MAX_NEGATIVE_FALSE_ACCEPT_RATE = 0.05
MIN_ACCEPTED_PRECISION = 0.95


def classify_outcome(
    is_supported: bool,
    top1_correct: bool | None,
    accepted: bool,
) -> str:
    """Classify the decision outcome for a single query."""
    if is_supported:
        if accepted:
            return "correct_accept" if top1_correct else "wrong_accept"
        else:
            return "abstain_correct_candidate" if top1_correct else "abstain_wrong_candidate"
    else:
        return "false_accept" if accepted else "correct_abstain"


def calculate_policy_metrics(
    queries: list[dict],
    similarity_threshold: float,
    margin_threshold: float,
) -> dict:
    """Compute acceptance, precision, coverage, false-acceptance, and abstention metrics."""
    correct_accept_count = 0
    wrong_accept_count = 0
    negative_false_accept_count = 0
    abstain_correct_candidate_count = 0
    abstain_wrong_candidate_count = 0
    correct_abstain_count = 0

    supported_queries = [q for q in queries if q.get("expected_document_id") is not None]
    negative_queries = [q for q in queries if q.get("expected_document_id") is None]
    supported_total = len(supported_queries)
    negative_total = len(negative_queries)
    total = len(queries)

    for q in queries:
        top1_sim = q.get("top1_similarity")
        margin = q.get("similarity_margin")
        is_supp = q.get("expected_document_id") is not None

        # Missing or nonfinite scores cannot pass thresholds
        accepted = (
            top1_sim is not None
            and margin is not None
            and math.isfinite(top1_sim)
            and math.isfinite(margin)
            and top1_sim >= similarity_threshold
            and margin >= margin_threshold
        )

        if is_supp:
            is_correct = bool(q.get("top1_correct"))
            if accepted:
                if is_correct:
                    correct_accept_count += 1
                else:
                    wrong_accept_count += 1
            else:
                if is_correct:
                    abstain_correct_candidate_count += 1
                else:
                    abstain_wrong_candidate_count += 1
        else:
            if accepted:
                negative_false_accept_count += 1
            else:
                correct_abstain_count += 1

    accepted_count = correct_accept_count + wrong_accept_count + negative_false_accept_count
    abstained_count = total - accepted_count

    accepted_precision = correct_accept_count / accepted_count if accepted_count > 0 else None
    supported_correct_coverage = correct_accept_count / supported_total if supported_total > 0 else None
    negative_false_accept_rate = negative_false_accept_count / negative_total if negative_total > 0 else None
    wrong_supported_accept_rate = wrong_accept_count / supported_total if supported_total > 0 else None
    overall_abstention_rate = abstained_count / total if total > 0 else None
    supported_accept_rate = (correct_accept_count + wrong_accept_count) / supported_total if supported_total > 0 else None
    negative_abstention_rate = correct_abstain_count / negative_total if negative_total > 0 else None

    return {
        "accepted_count": accepted_count,
        "abstained_count": abstained_count,
        "correct_accept_count": correct_accept_count,
        "wrong_accept_count": wrong_accept_count,
        "negative_false_accept_count": negative_false_accept_count,
        "supported_correct_accept_count": correct_accept_count,
        "abstain_correct_candidate_count": abstain_correct_candidate_count,
        "abstain_wrong_candidate_count": abstain_wrong_candidate_count,
        "correct_abstain_count": correct_abstain_count,
        "supported_total": supported_total,
        "negative_total": negative_total,
        "total": total,
        "accepted_precision": accepted_precision,
        "supported_correct_coverage": supported_correct_coverage,
        "negative_false_accept_rate": negative_false_accept_rate,
        "wrong_supported_accept_rate": wrong_supported_accept_rate,
        "overall_abstention_rate": overall_abstention_rate,
        "supported_accept_rate": supported_accept_rate,
        "negative_abstention_rate": negative_abstention_rate,
    }


def generate_threshold_candidates(queries: list[dict]) -> tuple[list[float], list[float]]:
    """Generate deterministic candidate thresholds from calibration scores and boundary extremes."""
    sims = {
        float(q["top1_similarity"])
        for q in queries
        if q.get("top1_similarity") is not None and math.isfinite(q["top1_similarity"])
    }
    margins = {
        float(q["similarity_margin"])
        for q in queries
        if q.get("similarity_margin") is not None and math.isfinite(q["similarity_margin"])
    }

    # Boundaries:
    # Similarity: -1.0 (accept all cosine), 0.0 (non-negative cosine), 1.0 (reject all)
    # Margin: 0.0 (accept all margins), 1.0 (reject all margins)
    sim_boundaries = {-1.0, 0.0, 1.0}
    margin_boundaries = {0.0, 1.0}

    return sorted(sims | sim_boundaries), sorted(margins | margin_boundaries)


def threshold_precision(val: float) -> int:
    """Number of significant decimal digits in standard representation."""
    text = str(val)
    if "e" in text.lower():
        return 20
    if "." in text:
        dec = text.split(".")[1].rstrip("0")
        return len(dec)
    return 0


def policy_simplicity_key(similarity_threshold: float, margin_threshold: float) -> tuple:
    """Simpler/lower-precision threshold representation key for deterministic tie-breaking."""
    prec_s = threshold_precision(similarity_threshold)
    prec_m = threshold_precision(margin_threshold)
    return (
        prec_s + prec_m,
        len(str(similarity_threshold)) + len(str(margin_threshold)),
        similarity_threshold,
        margin_threshold,
    )


def search_optimal_policy(
    queries: list[dict],
    candidate_similarities: list[float],
    candidate_margins: list[float],
    max_negative_false_accept_rate: float = MAX_NEGATIVE_FALSE_ACCEPT_RATE,
    min_accepted_precision: float = MIN_ACCEPTED_PRECISION,
) -> dict:
    """Search combinations of similarity and margin thresholds using the predeclared objective."""
    total_evaluated = len(candidate_similarities) * len(candidate_margins)
    qualifying_policies = []

    for s in candidate_similarities:
        for m in candidate_margins:
            metrics = calculate_policy_metrics(queries, s, m)
            prec = metrics["accepted_precision"]
            neg_rate = metrics["negative_false_accept_rate"]

            neg_ok = (neg_rate is None) or (neg_rate <= max_negative_false_accept_rate)
            if prec is not None and prec >= min_accepted_precision and neg_ok:
                qualifying_policies.append({
                    "similarity_threshold": s,
                    "margin_threshold": m,
                    "calibration_metrics": metrics,
                })

    search_meta = {
        "candidate_similarity_thresholds_count": len(candidate_similarities),
        "candidate_margin_thresholds_count": len(candidate_margins),
        "total_policies_evaluated": total_evaluated,
        "qualifying_policies_count": len(qualifying_policies),
    }

    if not qualifying_policies:
        return {
            "found": False,
            "similarity_threshold": None,
            "margin_threshold": None,
            "calibration_metrics": None,
            "candidate_search": search_meta,
        }

    # Predeclared policy objective & tie-breaking:
    # 1. Maximize supported_correct_coverage among qualifying policies
    # Tie breaking, in order:
    # 1. lower negative_false_accept_rate
    # 2. fewer wrong supported accepts (wrong_accept_count)
    # 3. higher accepted_precision
    # 4. higher supported_correct_coverage
    # 5. simpler/lower-precision threshold representation
    def policy_rank_key(p):
        metrics = p["calibration_metrics"]
        s = p["similarity_threshold"]
        margin = p["margin_threshold"]
        simplicity = policy_simplicity_key(s, margin)
        neg_rate_val = metrics["negative_false_accept_rate"] if metrics["negative_false_accept_rate"] is not None else 0.0
        return (
            metrics["supported_correct_coverage"],
            -neg_rate_val,
            -metrics["wrong_accept_count"],
            metrics["accepted_precision"],
            -simplicity[0],
            -simplicity[1],
            -s,
            -margin,
        )

    best_policy = max(qualifying_policies, key=policy_rank_key)
    return {
        "found": True,
        "similarity_threshold": best_policy["similarity_threshold"],
        "margin_threshold": best_policy["margin_threshold"],
        "calibration_metrics": best_policy["calibration_metrics"],
        "candidate_search": search_meta,
    }


def evaluate_calibration_queries(
    dataset: SemanticDataset,
    retriever: SemanticRetriever,
) -> list[dict]:
    """Execute ranked search for calibration queries only, keeping held-out queries untouched."""
    cal_queries = [q for q in dataset.queries if q.split == "calibration"]
    if len(cal_queries) != 96:
        raise ValueError(f"Expected exactly 96 calibration queries, got {len(cal_queries)}")

    rows = []
    for query in cal_queries:
        candidates, error = [], None
        try:
            results = retriever.search(query.query, n_results=7)
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
        top1_corr = (top1_doc == query.expected_document_id) if query.expected_document_id is not None else None

        row = {
            **query.model_dump(),
            "ranked_candidates": candidates,
            "error": error,
            **scores,
            "top1_correct": top1_corr,
        }
        rows.append(row)
    return rows


def annotate_queries_with_decision(
    queries: list[dict],
    similarity_threshold: float | None,
    margin_threshold: float | None,
) -> list[dict]:
    """Annotate evaluated calibration queries with the decision and outcome under a policy."""
    annotated = []
    for q in queries:
        top1_sim = q.get("top1_similarity")
        margin = q.get("similarity_margin")
        is_supp = q.get("expected_document_id") is not None
        top1_corr = q.get("top1_correct")

        accepted = (
            similarity_threshold is not None
            and margin_threshold is not None
            and top1_sim is not None
            and margin is not None
            and math.isfinite(top1_sim)
            and math.isfinite(margin)
            and top1_sim >= similarity_threshold
            and margin >= margin_threshold
        )
        decision = "accept" if accepted else "abstain"
        outcome = classify_outcome(is_supp, top1_corr, accepted)

        item = {
            **q,
            "decision": decision,
            "outcome": outcome,
        }
        annotated.append(item)
    return annotated


def calculate_calibration_score_distributions(rows: list[dict]) -> dict:
    """Compute score distributions for calibration split only."""
    supported = [r for r in rows if r.get("expected_document_id") is not None]
    return {
        "negative": {
            category: score_analysis([r for r in rows if r["category"] == category])
            for category in NEGATIVE_CATEGORIES
        },
        "supported": {
            "correct_top1": score_analysis([r for r in supported if r.get("top1_correct")]),
            "incorrect_top1": score_analysis([r for r in supported if not r.get("top1_correct")]),
        },
    }


def build_calibration_report(
    dataset_path: Path | str = DATASET_PATH,
    *,
    store=None,
    retriever=None,
    rows: list[dict] | None = None,
    max_negative_false_accept_rate: float = MAX_NEGATIVE_FALSE_ACCEPT_RATE,
    min_accepted_precision: float = MIN_ACCEPTED_PRECISION,
) -> dict:
    """Calibrate the conjunctive policy on calibration queries and build report without held-out data."""
    dataset_hash = dataset_sha256(dataset_path)
    dataset = load_dataset(dataset_path)

    store = store if store is not None else ChromaVectorStore()
    kb_hash = knowledge_base_sha256(store.kb_directory)

    if rows is None:
        if retriever is None:
            encoder = EmbeddingService(
                model_name=store.model_name,
                revision=store.model_revision or "",
                local_files_only=True,
            )
            retriever = SemanticRetriever(encoder, store, KnowledgeBaseLoader(store.kb_directory))
        rows = evaluate_calibration_queries(dataset, retriever)

    if dataset_sha256(dataset_path) != dataset_hash or knowledge_base_sha256(store.kb_directory) != kb_hash:
        raise ValueError("Evaluation inputs changed during execution; rerun with stable files")

    # Generate candidate thresholds from calibration cases only
    cand_sims, cand_margins = generate_threshold_candidates(rows)

    # Search for optimal policy
    search_result = search_optimal_policy(
        rows,
        cand_sims,
        cand_margins,
        max_negative_false_accept_rate=max_negative_false_accept_rate,
        min_accepted_precision=min_accepted_precision,
    )

    # Annotate queries with policy decision & outcome
    annotated_queries = annotate_queries_with_decision(
        rows,
        search_result["similarity_threshold"],
        search_result["margin_threshold"],
    )

    score_distributions = calculate_calibration_score_distributions(rows)

    report = {
        "calibration": {
            "name": "semantic_retrieval_calibration",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_version": dataset.dataset_version,
            "dataset_sha256": dataset_hash,
            "kb_sha256": kb_hash,
            "git_revision": git_revision(),
            "policy_type": "similarity_and_margin",
            "embedding_model": store.model_name,
            "model_revision": store.model_revision,
            "embedding_dimension": store.dimension,
            "normalized_embeddings": True,
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
            "chroma_collection_name": store.collection_name,
            "distance_metric": "cosine",
            "python_version": platform.python_version(),
            "calibration_query_count": len(rows),
            "calibration_supported_count": sum(1 for r in rows if r.get("expected_document_id") is not None),
            "calibration_negative_count": sum(1 for r in rows if r.get("expected_document_id") is None),
        },
        "selection_constraints": {
            "max_negative_false_accept_rate": max_negative_false_accept_rate,
            "min_accepted_precision": min_accepted_precision,
        },
        "candidate_search": search_result["candidate_search"],
        "selected_policy": {
            "found": search_result["found"],
            "similarity_threshold": search_result["similarity_threshold"],
            "margin_threshold": search_result["margin_threshold"],
            "calibration_metrics": search_result["calibration_metrics"],
        },
        "score_distributions": score_distributions,
        "calibration_queries": annotated_queries,
    }
    return report


def write_calibration_report(report: dict, path: Path | str = CALIBRATION_REPORT_PATH) -> None:
    """Serialize calibration report with deterministic formatting."""
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def main() -> int:
    try:
        report = build_calibration_report()
        write_calibration_report(report)
    except Exception as exc:
        print(f"Semantic calibration failed: {exc}")
        return 1

    policy = report["selected_policy"]
    if not policy["found"]:
        print("Semantic calibration completed: NO QUALIFYING POLICY FOUND under constraints.")
        print(f"Report: {CALIBRATION_REPORT_PATH}")
        return 0

    m = policy["calibration_metrics"]
    print(f"Semantic calibration selected policy: similarity >= {policy['similarity_threshold']}, margin >= {policy['margin_threshold']}")
    print(f"Accepted: {m['accepted_count']}/{m['total']} (Supported: {m['correct_accept_count'] + m['wrong_accept_count']}/{m['supported_total']}, Negative: {m['negative_false_accept_count']}/{m['negative_total']})")
    print(f"Precision: {m['accepted_precision']:.4f}; Coverage: {m['supported_correct_coverage']:.4f}; Neg False-Accept Rate: {m['negative_false_accept_rate']:.4f}; Wrong Accept Rate: {m['wrong_supported_accept_rate']:.4f}; Abstention Rate: {m['overall_abstention_rate']:.4f}")
    print(f"Report: {CALIBRATION_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
