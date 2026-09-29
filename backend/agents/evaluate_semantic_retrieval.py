"""Fixed-label ranked semantic evaluation, without acceptance or calibration."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import mean, median
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from agents.embedding_service import EMBEDDING_TEXT_VERSION, EmbeddingService
from agents.evaluate_keyword_retrieval import git_revision
from agents.knowledge_base import KnowledgeBaseLoader
from agents.knowledge_base_fingerprint import BACKEND_DIR, DOCUMENT_IDS, knowledge_base_sha256
from agents.keyword_retriever import KeywordRetriever
from agents.semantic_retriever import SemanticRetriever
from agents.vector_store import ChromaVectorStore


DATASET_PATH = BACKEND_DIR / "tests" / "fixtures" / "retrieval" / "semantic_queries.json"
REPORT_PATH = BACKEND_DIR / "evaluation_results" / "semantic_baseline.json"
SUPPORTED_CATEGORIES = ("direct_semantic", "paraphrase", "description", "supported_natural_language")
NEGATIVE_CATEGORIES = ("unsupported_medical", "unrelated")
SPLITS = ("calibration", "held_out")
CATEGORY_COUNTS = dict.fromkeys(SUPPORTED_CATEGORIES, 21) | {"unsupported_medical": 40, "unrelated": 20}
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SemanticQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: NonBlank
    query: str
    category: Literal["direct_semantic", "paraphrase", "description", "supported_natural_language", "unsupported_medical", "unrelated"]
    expected_document_id: str | None
    split: Literal["calibration", "held_out"]
    group_id: NonBlank

    @model_validator(mode="after")
    def valid_query_and_label(self):
        if not self.query.strip():
            raise ValueError("Query text must not be blank")
        if self.category in SUPPORTED_CATEGORIES:
            if self.expected_document_id not in DOCUMENT_IDS:
                raise ValueError("Supported categories require an approved document ID")
        elif self.expected_document_id is not None:
            raise ValueError("Negative categories require a null label")
        return self


class SemanticDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_version: NonBlank
    description: NonBlank
    queries: list[SemanticQuery] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids_and_separate_groups(self):
        if len({row.id for row in self.queries}) != len(self.queries):
            raise ValueError("Duplicate query IDs")
        groups = {}
        for row in self.queries:
            if row.group_id in groups and groups[row.group_id] != row.split:
                raise ValueError(f"Group leakage across splits: {row.group_id}")
            groups[row.group_id] = row.split
        return self


def validate_distribution(dataset: SemanticDataset) -> None:
    rows = dataset.queries
    if len(rows) != 144 or Counter(row.category for row in rows) != CATEGORY_COUNTS:
        raise ValueError("Production fixture requires 144 queries with exact category counts")
    if Counter(row.split for row in rows) != {"calibration": 96, "held_out": 48}:
        raise ValueError("Production split counts must be 96 calibration / 48 held_out")
    supported = [row for row in rows if row.expected_document_id is not None]
    for identifier in DOCUMENT_IDS:
        selected = [row for row in supported if row.expected_document_id == identifier]
        if len(selected) != 12 or Counter(row.split for row in selected) != {"calibration": 8, "held_out": 4}:
            raise ValueError(f"Per-test distribution must be 12 with 8/4 split: {identifier}")
        for category in SUPPORTED_CATEGORIES:
            cases = [row for row in selected if row.category == category]
            if len(cases) != 3 or Counter(row.split for row in cases) != {"calibration": 2, "held_out": 1}:
                raise ValueError(f"Category/test distribution must be 3 with 2/1 split: {identifier}/{category}")
    for category, counts in {"unsupported_medical": {"calibration": 28, "held_out": 12},
                             "unrelated": {"calibration": 12, "held_out": 8}}.items():
        if Counter(row.split for row in rows if row.category == category) != counts:
            raise ValueError(f"Negative split distribution is incorrect: {category}")


def load_dataset(path: Path | str = DATASET_PATH, *, production: bool = True) -> SemanticDataset:
    dataset = SemanticDataset.model_validate_json(Path(path).read_bytes())
    if production:
        validate_distribution(dataset)
    return dataset


def dataset_sha256(path: Path | str = DATASET_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _top_scores(candidates: list[dict]) -> dict:
    scores = {}
    for position in (1, 2):
        candidate = candidates[position - 1] if len(candidates) >= position else None
        for field in ("document_id", "distance", "similarity"):
            scores[f"top{position}_{field}"] = candidate[field] if candidate else None
    scores["similarity_margin"] = (scores["top1_similarity"] - scores["top2_similarity"]
                                    if len(candidates) >= 2 else None)
    return scores


def evaluate_queries(dataset: SemanticDataset, retriever: SemanticRetriever,
                     keyword_retriever: KeywordRetriever) -> list[dict]:
    rows = []
    for query in dataset.queries:
        candidates, error = [], None
        try:
            results = retriever.search(query.query, n_results=7)
            # A nonempty partial/invalid ranking cannot establish full MRR.
            if results and (len(results) != 7 or {r.document_id for r in results} != set(DOCUMENT_IDS)):
                raise ValueError("Evaluation requires all seven distinct approved candidates")
            if any(r.rank != rank or not math.isfinite(r.distance) or not math.isfinite(r.similarity)
                   for rank, r in enumerate(results, start=1)):
                raise ValueError("Invalid candidate rank or nonfinite score")
            candidates = [{"rank": r.rank, "document_id": r.document_id, "test_name": r.test_name,
                           "distance": r.distance, "similarity": r.similarity} for r in results]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        expected_rank = next((r["rank"] for r in candidates if r["document_id"] == query.expected_document_id), None)
        row = {**query.model_dump(), "ranked_candidates": candidates, "error": error, **_top_scores(candidates)}
        if query.expected_document_id is not None:
            row.update(expected_rank=expected_rank, top1_correct=expected_rank == 1,
                       hit_at_3=expected_rank is not None and expected_rank <= 3,
                       reciprocal_rank=1.0 / expected_rank if expected_rank else 0.0)
            try:
                document = keyword_retriever.retrieve(query.query)
                row["keyword"] = {"document_id": document.id if document else None, "error": None}
            except Exception as exc:
                row["keyword"] = {"document_id": None, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
    return rows


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def ranking_metrics(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(row["top1_correct"] for row in rows)
    hits = sum(row["hit_at_3"] for row in rows)
    reciprocal_sum = sum(row["reciprocal_rank"] for row in rows)
    return {"total": total, "supported_query_count": total, "top1_correct": correct,
            "top1_accuracy": _rate(correct, total), "hit_at_3_count": hits,
            "hit_at_3": _rate(hits, total), "recall_at_3": _rate(hits, total),
            "reciprocal_rank_sum": reciprocal_sum, "mrr": _rate(reciprocal_sum, total),
            "error_count": sum(row["error"] is not None for row in rows)}


def descriptive_statistics(values: list[float]) -> dict:
    return {"count": len(values), "min": min(values) if values else None,
            "max": max(values) if values else None, "mean": mean(values) if values else None,
            "median": median(values) if values else None}


def score_analysis(rows: list[dict]) -> dict:
    return {"query_count": len(rows), "error_count": sum(row["error"] is not None for row in rows),
            **{field: descriptive_statistics([row[field] for row in rows if row[field] is not None])
               for field in ("top1_similarity", "top1_distance", "similarity_margin")}}


def keyword_metrics(rows: list[dict]) -> dict:
    correct = sum(row["keyword"]["error"] is None and
                  row["keyword"]["document_id"] == row["expected_document_id"] for row in rows)
    answered = sum(row["keyword"]["error"] is None and row["keyword"]["document_id"] is not None for row in rows)
    return {"supported_query_count": len(rows), "correct_document_count": correct,
            "correct_document_rate": _rate(correct, len(rows)), "answered_count": answered,
            "answered_rate": _rate(answered, len(rows)),
            "error_count": sum(row["keyword"]["error"] is not None for row in rows)}


def calculate_metrics(rows: list[dict]) -> dict:
    supported = [row for row in rows if row["expected_document_id"] is not None]
    per_test = {identifier: ranking_metrics([row for row in supported if row["expected_document_id"] == identifier])
                for identifier in DOCUMENT_IDS}
    accuracies = [value["top1_accuracy"] for value in per_test.values()]
    return {
        "dataset_counts": {"total": len(rows), "supported": len(supported), "negative": len(rows) - len(supported),
                           "categories": dict(Counter(row["category"] for row in rows)),
                           "splits": {split: {"total": sum(row["split"] == split for row in rows),
                                     "supported": sum(row["split"] == split for row in supported),
                                     "negative": sum(row["split"] == split and row["expected_document_id"] is None for row in rows)}
                                      for split in SPLITS},
                           "semantic_error_count": sum(row["error"] is not None for row in rows)},
        "supported_metrics": {**ranking_metrics(supported),
            "macro_supported_test_top1_accuracy": mean(accuracies) if all(v is not None for v in accuracies) else None},
        "category_metrics": {category: ranking_metrics([row for row in supported if row["category"] == category])
                             for category in SUPPORTED_CATEGORIES},
        "per_test_metrics": per_test,
        "split_metrics": {split: ranking_metrics([row for row in supported if row["split"] == split]) for split in SPLITS},
        "negative_analysis": {category: score_analysis([row for row in rows if row["category"] == category])
                              for category in NEGATIVE_CATEGORIES},
        "supported_score_analysis": {
            "correct_top1": score_analysis([row for row in supported if row["top1_correct"]]),
            "incorrect_top1": score_analysis([row for row in supported if not row["top1_correct"]]),
        },
        # Split-specific scores avoid needing held-out observations for later calibration.
        "score_analysis_by_split": {split: {
            "negative": {category: score_analysis([row for row in rows if row["split"] == split and row["category"] == category])
                         for category in NEGATIVE_CATEGORIES},
            "supported_correct_top1": score_analysis([row for row in supported if row["split"] == split and row["top1_correct"]]),
            "supported_incorrect_top1": score_analysis([row for row in supported if row["split"] == split and not row["top1_correct"]]),
        } for split in SPLITS},
        "keyword_comparison": {**keyword_metrics(supported),
                               "splits": {split: keyword_metrics([row for row in supported if row["split"] == split]) for split in SPLITS}},
    }


def build_report(dataset_path: Path | str = DATASET_PATH, *, store=None, retriever=None,
                 keyword_retriever=None) -> dict:
    dataset_hash = dataset_sha256(dataset_path)
    dataset = load_dataset(dataset_path)
    store = store if store is not None else ChromaVectorStore()
    kb_hash = knowledge_base_sha256(store.kb_directory)
    if retriever is None:
        # Cached model only; the evaluation never downloads or rebuilds an index.
        encoder = EmbeddingService(model_name=store.model_name, revision=store.model_revision or "", local_files_only=True)
        retriever = SemanticRetriever(encoder, store, KnowledgeBaseLoader(store.kb_directory))
    keyword_retriever = keyword_retriever if keyword_retriever is not None else KeywordRetriever(KnowledgeBaseLoader(store.kb_directory))
    rows = evaluate_queries(dataset, retriever, keyword_retriever)
    if dataset_sha256(dataset_path) != dataset_hash or knowledge_base_sha256(store.kb_directory) != kb_hash:
        raise ValueError("Evaluation inputs changed during execution; rerun with stable files")
    return {"evaluation": {
        "name": "semantic_retrieval_baseline", "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset.dataset_version, "dataset_sha256": dataset_hash,
        "knowledge_base_sha256": kb_hash, "git_revision": git_revision(),
        "embedding_model": store.model_name, "model_revision": store.model_revision,
        "embedding_dimension": store.dimension, "normalized_embeddings": True,
        "embedding_text_version": EMBEDDING_TEXT_VERSION, "chroma_collection_name": store.collection_name,
        "distance_metric": "cosine", "python_version": platform.python_version(), "requested_candidates": 7,
        "acceptance_policy": "none; ranked search only",
    }, **calculate_metrics(rows), "queries": rows}


def write_report(report: dict, path: Path | str = REPORT_PATH) -> None:
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def main() -> int:
    try:
        report = build_report()
        write_report(report)
    except Exception as exc:
        print(f"Semantic evaluation failed: {exc}")
        return 1
    metrics = report["supported_metrics"]
    print(f"Supported: {metrics['top1_correct']}/{metrics['total']}; "
          f"Top-1: {metrics['top1_accuracy']:.4f}; Hit/Recall@3: {metrics['hit_at_3']:.4f}; MRR: {metrics['mrr']:.4f}")
    print(f"Semantic errors: {report['dataset_counts']['semantic_error_count']}; Report: {REPORT_PATH}")
    return int(bool(report["dataset_counts"]["semantic_error_count"] or report["keyword_comparison"]["error_count"]))


if __name__ == "__main__":
    raise SystemExit(main())
