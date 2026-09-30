"""Offline tests for SemanticRetriever.retrieve() and acceptance-policy types.

All 16 test cases (A-P) mandated by Step 12.5. No MiniLM, no Chroma, no
network access. search() behavior and integrity tests remain in
test_semantic_retriever.py.
"""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from agents import semantic_retriever as module
from agents.knowledge_base import KnowledgeDocument
from agents.knowledge_base_fingerprint import document_sha256
from agents.semantic_retriever import (
    FROZEN_POLICY,
    SemanticAcceptancePolicy,
    SemanticRetrievalDecision,
    SemanticRetriever,
    SemanticRetrievalIntegrityError,
    SemanticSearchResult,
)
from agents.vector_store import VectorIndexStaleError, VectorIndexNotBuiltError, VectorSearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_doc(i: int) -> KnowledgeDocument:
    return KnowledgeDocument(
        id=str(i),
        test_name=f"Test {i}",
        aliases=[f"Alias {i}"],
        report_type="CBC",
        title="Educational fixture",
        definition="Synthetic definition",
        what_it_measures="Synthetic measurement",
        general_information="Synthetic context",
        source={
            "publisher": "Fixture",
            "title": "Fixture source",
            "url": "https://example.org",
            "accessed_date": "2026-09-29",
        },
    )


def _make_result(doc: KnowledgeDocument, distance: float, rank: int) -> SemanticSearchResult:
    return SemanticSearchResult(
        document_id=doc.id,
        document=doc.model_copy(deep=True),
        distance=distance,
        rank=rank,
    )


@pytest.fixture()
def retriever_with_search():
    """Return a SemanticRetriever whose search() is patched to a controllable mock."""
    encoder = SimpleNamespace(model_name="fake", revision=None, encode_text=Mock(return_value=[1.0, 0.0]))
    store = SimpleNamespace(
        model_name="fake",
        model_revision=None,
        dimension=2,
        kb_directory="unused",
        query=Mock(return_value=[]),
    )
    loader = SimpleNamespace(load_all=Mock(return_value=[]))
    retriever = SemanticRetriever(encoder, store, loader)
    return retriever, store


# ---------------------------------------------------------------------------
# A. Accepted when similarity AND margin both pass
# ---------------------------------------------------------------------------

def test_a_accepted_when_both_thresholds_pass(retriever_with_search):
    """A: accepted=True when top1.similarity >= 0.0 AND margin >= 0.22541916370391846."""
    retriever, _ = retriever_with_search
    docs = [_make_doc(0), _make_doc(1)]
    # top1.similarity = 1 - 0.1 = 0.9; top2.similarity = 1 - 0.6 = 0.4; margin = 0.5 (passes)
    top1 = _make_result(docs[0], distance=0.1, rank=1)
    top2 = _make_result(docs[1], distance=0.6, rank=2)
    with patch.object(retriever, "search", return_value=[top1, top2]) as mock_search:
        decision = retriever.retrieve("hemoglobin measurement")
    mock_search.assert_called_once_with("hemoglobin measurement", n_results=2)
    assert decision.accepted is True
    assert decision.document is not None
    assert decision.document.id == docs[0].id


# ---------------------------------------------------------------------------
# B. Abstain when similarity fails (below threshold)
# ---------------------------------------------------------------------------

def test_b_abstain_when_similarity_fails(retriever_with_search):
    """B: abstain when top1.similarity < similarity_threshold (0.0)."""
    retriever, _ = retriever_with_search
    docs = [_make_doc(0), _make_doc(1)]
    # distance=1.05 => similarity=-0.05 (below 0.0); margin=1.0 (would pass alone)
    top1 = _make_result(docs[0], distance=1.05, rank=1)
    top2 = _make_result(docs[1], distance=2.05, rank=2)
    with patch.object(retriever, "search", return_value=[top1, top2]):
        decision = retriever.retrieve("query")
    assert decision.accepted is False
    assert decision.document is None


# ---------------------------------------------------------------------------
# C. Abstain when margin fails
# ---------------------------------------------------------------------------

def test_c_abstain_when_margin_fails(retriever_with_search):
    """C: abstain when margin < margin_threshold even if similarity passes."""
    retriever, _ = retriever_with_search
    docs = [_make_doc(0), _make_doc(1)]
    # similarity = 0.6 (passes); margin = 0.05 (below 0.22541916370391846)
    top1 = _make_result(docs[0], distance=0.4, rank=1)
    top2 = _make_result(docs[1], distance=0.45, rank=2)
    with patch.object(retriever, "search", return_value=[top1, top2]):
        decision = retriever.retrieve("query")
    assert decision.accepted is False
    assert decision.document is None


# ---------------------------------------------------------------------------
# D. Equality at similarity threshold passes
# ---------------------------------------------------------------------------

def test_d_equality_at_similarity_threshold_passes(retriever_with_search):
    """D: similarity == 0.0 (distance==1.0) is accepted if margin also passes."""
    retriever, _ = retriever_with_search
    docs = [_make_doc(0), _make_doc(1)]
    # similarity = 1 - 1.0 = 0.0 (exactly at threshold)
    # margin = 0.0 - (1 - 1.3) = 0.0 - (-0.3) = 0.3 (passes margin)
    top1 = _make_result(docs[0], distance=1.0, rank=1)
    top2 = _make_result(docs[1], distance=1.3, rank=2)
    with patch.object(retriever, "search", return_value=[top1, top2]):
        decision = retriever.retrieve("query")
    assert top1.similarity == 0.0
    margin = top1.similarity - top2.similarity
    assert margin > FROZEN_POLICY.margin_threshold
    assert decision.accepted is True


# ---------------------------------------------------------------------------
# E. Equality at margin threshold passes
# ---------------------------------------------------------------------------

def test_e_equality_at_margin_threshold_passes(retriever_with_search):
    """E: margin == margin_threshold exactly is accepted."""
    retriever, _ = retriever_with_search
    docs = [_make_doc(0), _make_doc(1)]
    # Both values are binary-exact: top1 similarity equals threshold, top2 is zero.
    margin_threshold = FROZEN_POLICY.margin_threshold
    top1_sim = margin_threshold
    top2_sim = top1_sim - margin_threshold
    top1 = _make_result(docs[0], distance=1.0 - top1_sim, rank=1)
    top2 = _make_result(docs[1], distance=1.0 - top2_sim, rank=2)
    with patch.object(retriever, "search", return_value=[top1, top2]):
        decision = retriever.retrieve("query")
    actual_margin = top1.similarity - top2.similarity
    assert actual_margin == margin_threshold
    assert decision.accepted is True


# ---------------------------------------------------------------------------
# F. Zero search results -> abstain
# ---------------------------------------------------------------------------

def test_f_zero_results_abstain(retriever_with_search):
    """F: zero candidates -> abstain with top_candidate=None, runner_up=None."""
    retriever, _ = retriever_with_search
    with patch.object(retriever, "search", return_value=[]):
        decision = retriever.retrieve("query")
    assert decision.accepted is False
    assert decision.document is None
    assert decision.top_candidate is None
    assert decision.runner_up is None
    assert decision.similarity_margin is None


# ---------------------------------------------------------------------------
# G. One search result -> abstain
# ---------------------------------------------------------------------------

def test_g_one_result_abstains(retriever_with_search):
    """G: single candidate -> abstain; top_candidate preserved, runner_up=None."""
    retriever, _ = retriever_with_search
    doc = _make_doc(0)
    top1 = _make_result(doc, distance=0.1, rank=1)
    with patch.object(retriever, "search", return_value=[top1]):
        decision = retriever.retrieve("query")
    assert decision.accepted is False
    assert decision.document is None
    assert decision.top_candidate is top1
    assert decision.runner_up is None
    assert decision.similarity_margin is None


# ---------------------------------------------------------------------------
# H. Accepted document is Top-1 authoritative KnowledgeDocument
# ---------------------------------------------------------------------------

def test_h_accepted_document_is_top1_authoritative(retriever_with_search):
    """H: decision.document is the KnowledgeDocument carried by the top-1 result."""
    retriever, _ = retriever_with_search
    docs = [_make_doc(0), _make_doc(1)]
    top1 = _make_result(docs[0], distance=0.1, rank=1)
    top2 = _make_result(docs[1], distance=0.6, rank=2)
    with patch.object(retriever, "search", return_value=[top1, top2]):
        decision = retriever.retrieve("query")
    assert decision.accepted is True
    assert decision.document == top1.document
    assert decision.document is top1.document


# ---------------------------------------------------------------------------
# I. top_candidate and runner_up retained in decision
# ---------------------------------------------------------------------------

def test_i_candidates_retained_in_abstaining_decision(retriever_with_search):
    """I: when abstaining with two candidates, top_candidate and runner_up are preserved."""
    retriever, _ = retriever_with_search
    docs = [_make_doc(0), _make_doc(1)]
    # margin too small -> abstain
    top1 = _make_result(docs[0], distance=0.4, rank=1)
    top2 = _make_result(docs[1], distance=0.45, rank=2)
    with patch.object(retriever, "search", return_value=[top1, top2]):
        decision = retriever.retrieve("query")
    assert decision.accepted is False
    assert decision.top_candidate is top1
    assert decision.runner_up is top2


def test_i_candidates_retained_in_accepted_decision(retriever_with_search):
    """I: when accepted, top_candidate and runner_up are also present."""
    retriever, _ = retriever_with_search
    docs = [_make_doc(0), _make_doc(1)]
    top1 = _make_result(docs[0], distance=0.1, rank=1)
    top2 = _make_result(docs[1], distance=0.6, rank=2)
    with patch.object(retriever, "search", return_value=[top1, top2]):
        decision = retriever.retrieve("query")
    assert decision.top_candidate is top1
    assert decision.runner_up is top2


# ---------------------------------------------------------------------------
# J. Exact similarity margin calculation
# ---------------------------------------------------------------------------

def test_j_exact_margin_calculation(retriever_with_search):
    """J: similarity_margin == top1.similarity - top2.similarity exactly."""
    retriever, _ = retriever_with_search
    docs = [_make_doc(0), _make_doc(1)]
    # Deliberate irrational distances to exercise float arithmetic
    top1 = _make_result(docs[0], distance=0.3, rank=1)
    top2 = _make_result(docs[1], distance=0.7, rank=2)
    with patch.object(retriever, "search", return_value=[top1, top2]):
        decision = retriever.retrieve("query")
    expected_margin = (1.0 - 0.3) - (1.0 - 0.7)
    assert decision.similarity_margin == expected_margin


# ---------------------------------------------------------------------------
# K. search() called with n_results=2
# ---------------------------------------------------------------------------

def test_k_search_called_with_n_results_2(retriever_with_search):
    """K: retrieve() calls search(query, n_results=2) exactly once."""
    retriever, _ = retriever_with_search
    docs = [_make_doc(0), _make_doc(1)]
    top1 = _make_result(docs[0], distance=0.1, rank=1)
    top2 = _make_result(docs[1], distance=0.6, rank=2)
    with patch.object(retriever, "search", return_value=[top1, top2]) as mock_search:
        retriever.retrieve("my query")
    mock_search.assert_called_once_with("my query", n_results=2)


# ---------------------------------------------------------------------------
# L. search() exceptions propagate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exc", [
    VectorIndexStaleError("stale"),
    VectorIndexNotBuiltError("missing"),
    RuntimeError("encoder down"),
    SemanticRetrievalIntegrityError("fingerprint mismatch"),
    ValueError("bad vector"),
])
def test_l_search_exceptions_propagate(retriever_with_search, exc):
    """L: infrastructure/integrity exceptions from search() are not caught by retrieve()."""
    retriever, _ = retriever_with_search
    with patch.object(retriever, "search", side_effect=exc):
        with pytest.raises(type(exc)) as caught:
            retriever.retrieve("query")
    assert caught.value is exc


# ---------------------------------------------------------------------------
# M. search() behavior remains unchanged
# ---------------------------------------------------------------------------

def test_m_search_still_uses_default_n_results_3():
    """M: search() default is still n_results=3; retrieve() does not change its signature."""
    import inspect
    sig = inspect.signature(SemanticRetriever.search)
    assert sig.parameters["n_results"].default == 3


def test_m_search_returns_ranked_list(retriever_with_search):
    """M: search() still returns a plain list of SemanticSearchResult unchanged."""
    retriever, store = retriever_with_search
    doc = _make_doc(0)
    candidate = VectorSearchResult(doc.id, 0.2, {"document_sha256": document_sha256(doc)}, "text")
    store.query.return_value = [candidate]
    loader_mock = SimpleNamespace(load_all=Mock(return_value=[doc]))
    retriever._loader = loader_mock
    results = retriever.search("text", n_results=1)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], SemanticSearchResult)
    assert results[0].rank == 1


# ---------------------------------------------------------------------------
# N. Frozen policy constants match calibration-selected values exactly
# ---------------------------------------------------------------------------

def test_n_frozen_policy_constants_exact():
    """N: FROZEN_POLICY thresholds equal the calibration-selected values exactly."""
    assert FROZEN_POLICY.similarity_threshold == 0.0
    assert FROZEN_POLICY.margin_threshold == 0.22541916370391846


def test_n_policy_class_defaults_exact():
    """N: SemanticAcceptancePolicy() default construction yields exact calibration values."""
    policy = SemanticAcceptancePolicy()
    assert policy.similarity_threshold == 0.0
    assert policy.margin_threshold == 0.22541916370391846


# ---------------------------------------------------------------------------
# O. No Chroma writes/rebuilds occur
# ---------------------------------------------------------------------------

def test_o_no_chroma_writes_during_retrieve(retriever_with_search):
    """O: retrieve() performs no writes, rebuilds, or deletions on the vector store."""
    retriever, store = retriever_with_search
    for write_op in ("rebuild", "delete_collection", "add", "upsert", "modify"):
        setattr(store, write_op, Mock(side_effect=AssertionError(f"Write forbidden: {write_op}")))
    docs = [_make_doc(0), _make_doc(1)]
    top1 = _make_result(docs[0], distance=0.1, rank=1)
    top2 = _make_result(docs[1], distance=0.6, rank=2)
    with patch.object(retriever, "search", return_value=[top1, top2]):
        retriever.retrieve("query")
    for write_op in ("rebuild", "delete_collection", "add", "upsert", "modify"):
        getattr(store, write_op).assert_not_called()


# ---------------------------------------------------------------------------
# P. Decision dataclasses are immutable (frozen)
# ---------------------------------------------------------------------------

def test_p_semantic_acceptance_policy_is_frozen():
    """P: SemanticAcceptancePolicy is a frozen dataclass — fields cannot be reassigned."""
    policy = SemanticAcceptancePolicy()
    with pytest.raises(FrozenInstanceError):
        policy.similarity_threshold = 0.5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        policy.margin_threshold = 0.0  # type: ignore[misc]


def test_p_semantic_retrieval_decision_is_frozen(retriever_with_search):
    """P: SemanticRetrievalDecision is a frozen dataclass — fields cannot be reassigned."""
    retriever, _ = retriever_with_search
    with patch.object(retriever, "search", return_value=[]):
        decision = retriever.retrieve("query")
    with pytest.raises(FrozenInstanceError):
        decision.accepted = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        decision.document = None  # type: ignore[misc]


def test_p_semantic_search_result_still_frozen():
    """P: SemanticSearchResult (unchanged) also remains frozen."""
    doc = _make_doc(0)
    result = SemanticSearchResult(document_id=doc.id, document=doc, distance=0.3, rank=1)
    with pytest.raises(FrozenInstanceError):
        result.rank = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Extra: exact inclusive comparison boundaries
# ---------------------------------------------------------------------------

def test_policy_rejects_nan_margin():
    """accepts() returns False when margin is NaN."""
    import math
    assert FROZEN_POLICY.accepts(0.5, float("nan")) is False


def test_policy_accepts_zero_similarity_with_large_margin():
    """Boundary: similarity=0.0 and margin > threshold => True."""
    assert FROZEN_POLICY.accepts(0.0, FROZEN_POLICY.margin_threshold) is True


def test_policy_rejects_margin_one_ulp_below_threshold():
    """margin just below margin_threshold => False."""
    import math
    just_below = math.nextafter(FROZEN_POLICY.margin_threshold, 0.0)
    assert FROZEN_POLICY.accepts(0.9, just_below) is False
