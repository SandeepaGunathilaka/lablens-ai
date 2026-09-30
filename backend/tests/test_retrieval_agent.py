"""Offline orchestration contracts: mock hybrid retrieval, no model or index."""

from unittest.mock import Mock, call

import pytest
from pydantic import ValidationError

from agents import retrieval_agent as module
from agents.hybrid_retriever import HybridRetriever, HybridRetrievalResult
from agents.knowledge_base import KnowledgeBaseLoader, KnowledgeBaseValidationError
from agents.retrieval_agent import MedicalRetrievalAgent
from agents.retrieval_models import RetrievalRequest, RetrievalResponse
from agents.semantic_retriever import (
    SemanticRetrievalDecision, SemanticRetrievalIntegrityError, SemanticSearchResult,
)
from agents.vector_store import VectorIndexNotBuiltError, VectorIndexStaleError


IDS = {"task_id": " task-001 ", "report_id": "report/002", "user_id": "user-003"}


@pytest.fixture
def document():
    return KnowledgeBaseLoader().get_by_test_name("Hemoglobin")


@pytest.fixture(autouse=True)
def forbid_external_work(monkeypatch):
    """Fail if orchestration bypasses the injected hybrid or accesses storage."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Orchestration must only call the injected hybrid")

    monkeypatch.setattr(module, "HybridRetriever", Mock(side_effect=forbidden))
    for target in (
        "agents.keyword_retriever.KeywordRetriever.retrieve",
        "agents.semantic_retriever.SemanticRetriever.retrieve",
        "agents.semantic_retriever.SemanticRetriever.search",
        "agents.embedding_service.EmbeddingService.__init__",
        "agents.vector_store.ChromaVectorStore.__init__",
        "pymongo.MongoClient.__init__",
        "database.get_users_collection",
        "database.get_audit_logs_collection",
        "builtins.open",
        "pathlib.Path.write_text",
        "pathlib.Path.write_bytes",
    ):
        monkeypatch.setattr(target, forbidden)


def hybrid_with(*results):
    hybrid = Mock(spec_set=HybridRetriever)
    hybrid.retrieve.side_effect = results
    return hybrid


@pytest.mark.parametrize("method,query", [
    ("keyword", "Hgb"),
    ("semantic", "oxygen carrying protein"),
])
def test_success_preserves_curated_content_provenance_and_ids(document, method, query):
    top = SemanticSearchResult(document.id, document, 0.1234567, 1)
    decision = SemanticRetrievalDecision(True, document, top, None, 0.5)
    hybrid = hybrid_with(HybridRetrievalResult(document, True, method, decision))
    request = RetrievalRequest(**IDS, test_names=[query])
    request_before = request.model_dump()
    document_before = document.model_dump()

    response = MedicalRetrievalAgent(hybrid).retrieve(request)

    assert isinstance(response, RetrievalResponse)
    assert {key: getattr(response, key) for key in IDS} == IDS
    assert len(response.results) == 1
    result = response.results[0]
    assert result.test_name == query
    assert result.found is True
    assert len(result.matches) == 1
    match = result.matches[0]
    for label, value in (
        ("Test", document.test_name),
        ("Title", document.title),
        ("Definition", document.definition),
        ("What it measures", document.what_it_measures),
        ("General information", document.general_information),
        ("Source publisher", document.source.publisher),
        ("Source accessed date", document.source.accessed_date),
    ):
        assert f"{label}: {value}" in match.information.split("\n\n")
    assert len(match.sources) == 1
    assert match.sources[0].title == document.source.title
    assert str(match.sources[0].url) == document.source.url
    assert request.model_dump() == request_before
    assert document.model_dump() == document_before
    assert hybrid.mock_calls == [call.retrieve(query)]
    payload = response.model_dump_json()
    for internal_field in ("confidence", "similarity", "embedding", "aliases", "semantic_decision", "method"):
        assert f'"{internal_field}"' not in payload
    assert "0.1234567" not in payload
    assert RetrievalResponse.model_validate_json(payload) == response


def test_abstention_never_exposes_top_candidate(document):
    top = SemanticSearchResult(document.id, document, 0.01, 1)
    decision = SemanticRetrievalDecision(False, None, top, None, None)
    hybrid = hybrid_with(HybridRetrievalResult(None, False, "none", decision))
    response = MedicalRetrievalAgent(hybrid).retrieve(
        RetrievalRequest(**IDS, test_names=["unsupported query"])
    )
    assert response.model_dump() == {
        **IDS,
        "results": [{"test_name": "unsupported query", "found": False, "matches": []}],
    }
    assert document.definition not in response.model_dump_json()
    assert hybrid.mock_calls == [call.retrieve("unsupported query")]


def test_mixed_results_order_duplicates_and_independent_positions(document):
    ldl = KnowledgeBaseLoader().get_by_test_name("LDL")
    names = ["Hgb", "LDL-C", "unknown", "Hgb", "oxygen carrier"]
    hybrid = hybrid_with(
        HybridRetrievalResult(document, True, "keyword", None),
        HybridRetrievalResult(ldl, True, "keyword", None),
        HybridRetrievalResult(None, False, "none", None),
        HybridRetrievalResult(document, True, "keyword", None),
        HybridRetrievalResult(document, True, "semantic", None),
    )
    request = RetrievalRequest(**IDS, test_names=names)
    before = request.model_dump()
    response = MedicalRetrievalAgent(hybrid).retrieve(request)
    assert [result.test_name for result in response.results] == names
    assert [result.found for result in response.results] == [True, True, False, True, True]
    assert ldl.definition in response.results[1].matches[0].information
    assert response.results[0] == response.results[3]
    assert response.results[0] is not response.results[3]
    assert hybrid.mock_calls == [call.retrieve(name) for name in names]
    assert request.model_dump() == before
    assert {key: getattr(response, key) for key in IDS} == IDS


@pytest.mark.parametrize("error", [
    RuntimeError("embedding model failure"),
    ValueError("Query must be a nonblank string"),
    VectorIndexStaleError("stale index"),
    VectorIndexNotBuiltError("missing index"),
    KnowledgeBaseValidationError(["invalid KB"]),
    SemanticRetrievalIntegrityError("fingerprint mismatch"),
])
def test_errors_propagate_unchanged(error):
    hybrid = Mock(spec_set=HybridRetriever)
    hybrid.retrieve.side_effect = error
    with pytest.raises(type(error)) as caught:
        MedicalRetrievalAgent(hybrid).retrieve(RetrievalRequest(**IDS, test_names=["query"]))
    assert caught.value is error
    assert hybrid.mock_calls == [call.retrieve("query")]


def test_invalid_request_is_rejected_by_existing_model():
    with pytest.raises(ValidationError):
        RetrievalRequest(**IDS, test_names=[" "])


def test_inconsistent_success_is_an_error():
    hybrid = hybrid_with(HybridRetrievalResult(None, True, "keyword", None))
    with pytest.raises(ValueError, match="KnowledgeDocument"):
        MedicalRetrievalAgent(hybrid).retrieve(RetrievalRequest(**IDS, test_names=["Hgb"]))


def test_default_constructor_uses_hybrid_factory(monkeypatch):
    hybrid = hybrid_with(HybridRetrievalResult(None, False, "none", None))
    factory = Mock(return_value=hybrid)
    monkeypatch.setattr(module, "HybridRetriever", factory)
    agent = MedicalRetrievalAgent()
    factory.assert_called_once_with()
    response = agent.retrieve(RetrievalRequest(**IDS, test_names=["query"]))
    assert response.results[0].found is False
    assert hybrid.mock_calls == [call.retrieve("query")]


def test_repeated_requests_do_not_retain_results_or_identifiers(document):
    hybrid = hybrid_with(
        HybridRetrievalResult(document, True, "keyword", None),
        HybridRetrievalResult(None, False, "none", None),
    )
    agent = MedicalRetrievalAgent(hybrid)
    first = agent.retrieve(RetrievalRequest(**IDS, test_names=["Hgb"]))
    second_ids = {key: "second-" + value for key, value in IDS.items()}
    second = agent.retrieve(RetrievalRequest(**second_ids, test_names=["unknown"]))
    assert first.results[0].found is True
    assert second.model_dump() == {
        **second_ids, "results": [{"test_name": "unknown", "found": False, "matches": []}],
    }
    assert vars(agent) == {"_hybrid_retriever": hybrid}
