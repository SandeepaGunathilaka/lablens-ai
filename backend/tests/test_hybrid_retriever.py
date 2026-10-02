"""Offline hybrid routing tests; no model or persistent vector client."""

from dataclasses import FrozenInstanceError
from unittest.mock import Mock, call

import pytest

from agents import hybrid_retriever as module
from agents.hybrid_retriever import HybridRetriever, HybridRetrievalResult
from agents.knowledge_base import KnowledgeBaseLoader, KnowledgeBaseValidationError
from agents.keyword_retriever import KeywordRetriever
from agents.semantic_retriever import SemanticRetrievalDecision, SemanticSearchResult, SemanticRetrievalIntegrityError
from agents.vector_store import VectorIndexNotBuiltError, VectorIndexStaleError


@pytest.fixture
def document():
    return KnowledgeBaseLoader().get_by_test_name("Hemoglobin")


@pytest.fixture(autouse=True)
def no_default_semantics(monkeypatch):
    factory = Mock(side_effect=AssertionError("Real semantic construction forbidden"))
    monkeypatch.setattr(module, "SemanticRetriever", factory)
    return factory


@pytest.mark.parametrize("query,expected", [
    ("Hemoglobin", "hemoglobin"), ("WBC", "wbc"), ("Platelets", "platelets"),
    ("HDL", "hdl"), ("LDL", "ldl"), ("Triglycerides", "triglycerides"),
    ("Total Cholesterol", "total_cholesterol"), ("Hgb", "hemoglobin"),
    ("PLT", "platelets"), ("LDL-C", "ldl"), ("White Blood Cell Count", "wbc"),
    ("TC", "total_cholesterol"), ("  white   blood cell COUNT  ", "wbc"),
])
def test_real_keyword_precedence_without_semantic_dependency(query, expected, no_default_semantics):
    result = HybridRetriever().retrieve(query)
    assert result.found is True
    assert result.method == "keyword"
    assert result.document.id == expected
    assert result.semantic_decision is None
    no_default_semantics.assert_not_called()


def test_keyword_document_identity_and_no_semantic_call(document):
    before = document.model_dump()
    keyword = Mock(retrieve=Mock(return_value=document))
    semantic = Mock(retrieve=Mock(side_effect=AssertionError("Keyword must win")))
    result = HybridRetriever(keyword, semantic).retrieve("  Hgb  ")
    keyword.retrieve.assert_called_once_with("  Hgb  ")
    semantic.retrieve.assert_not_called()
    assert result.document is document
    assert document.model_dump() == before
    assert result == HybridRetrievalResult(document, True, "keyword", None)


@pytest.mark.parametrize("accepted", [True, False])
def test_semantic_fallback_retains_decision_and_documents(document, accepted):
    before = document.model_dump()
    top = SemanticSearchResult(document.id, document, .1, 1)
    decision = SemanticRetrievalDecision(accepted, document if accepted else None, top, None, None)
    keyword = Mock(retrieve=Mock(return_value=None))
    semantic = Mock(retrieve=Mock(return_value=decision))
    calls = Mock()
    calls.attach_mock(keyword.retrieve, "keyword")
    calls.attach_mock(semantic.retrieve, "semantic")
    query = "  Oxygen  carrying protein in red blood cells  "
    result = HybridRetriever(keyword, semantic).retrieve(query)
    assert calls.mock_calls == [call.keyword(query), call.semantic(query)]
    assert result.found is accepted
    assert result.method == ("semantic" if accepted else "none")
    assert result.document is (document if accepted else None)
    assert result.semantic_decision is decision
    assert document.model_dump() == before


def test_real_keyword_miss_uses_semantic_retrieve_once(document):
    decision = SemanticRetrievalDecision(True, document, None, None, .4)
    semantic = Mock(retrieve=Mock(return_value=decision))
    query = "oxygen carrying protein in red blood cells"
    result = HybridRetriever(KeywordRetriever(), semantic).retrieve(query)
    semantic.retrieve.assert_called_once_with(query)
    semantic.search.assert_not_called()
    assert result.method == "semantic"


@pytest.mark.parametrize("query", ["", " \t\n", None, 1, True, [], {}])
def test_invalid_input_before_retrievers(query):
    keyword, semantic = Mock(), Mock()
    with pytest.raises(ValueError, match="nonblank string"):
        HybridRetriever(keyword, semantic).retrieve(query)
    assert keyword.mock_calls == semantic.mock_calls == []


def test_keyword_error_propagates_without_fallback():
    error = KnowledgeBaseValidationError(["synthetic invalid KB"])
    keyword = Mock(retrieve=Mock(side_effect=error))
    semantic = Mock()
    with pytest.raises(KnowledgeBaseValidationError) as caught:
        HybridRetriever(keyword, semantic).retrieve("Hgb")
    assert caught.value is error
    assert semantic.mock_calls == []


@pytest.mark.parametrize("error", [VectorIndexNotBuiltError("missing"), VectorIndexStaleError("stale"),
    RuntimeError("encoding failed"), SemanticRetrievalIntegrityError("fingerprint mismatch"),
    KnowledgeBaseValidationError(["invalid KB"])])
def test_semantic_errors_propagate(error):
    keyword = Mock(retrieve=Mock(return_value=None))
    semantic = Mock(retrieve=Mock(side_effect=error))
    with pytest.raises(type(error)) as caught:
        HybridRetriever(keyword, semantic).retrieve("concept query")
    assert caught.value is error
    semantic.retrieve.assert_called_once_with("concept query")


def test_default_semantic_is_lazy_and_reused(monkeypatch):
    decision = SemanticRetrievalDecision(False, None, None, None, None)
    semantic = Mock(retrieve=Mock(return_value=decision))
    factory = Mock(return_value=semantic)
    monkeypatch.setattr(module, "SemanticRetriever", factory)
    keyword = Mock(retrieve=Mock(return_value=None))
    retriever = HybridRetriever(keyword)
    factory.assert_not_called()
    retriever.retrieve("first")
    retriever.retrieve("second")
    factory.assert_called_once_with()
    assert semantic.retrieve.call_args_list == [call("first"), call("second")]


@pytest.mark.parametrize("field,value", [("document", None), ("found", False), ("method", "none"), ("semantic_decision", None)])
def test_result_is_frozen(document, field, value):
    result = HybridRetrievalResult(document, True, "keyword", None)
    with pytest.raises(FrozenInstanceError):
        setattr(result, field, value)


@pytest.mark.parametrize("keyword_match", [True, False])
def test_no_writes_logging_or_query_history(document, keyword_match, caplog, capsys):
    keyword = Mock(retrieve=Mock(return_value=document if keyword_match else None))
    semantic = Mock(retrieve=Mock(return_value=SemanticRetrievalDecision(False,None,None,None,None)))
    retriever = HybridRetriever(keyword, semantic)
    initial_state = dict(vars(retriever))
    retriever.retrieve("private synthetic query")
    assert vars(retriever) == initial_state
    assert keyword.mock_calls == [call.retrieve("private synthetic query")]
    assert semantic.mock_calls == ([] if keyword_match else [call.retrieve("private synthetic query")])
    assert caplog.text == ""
    output = capsys.readouterr()
    assert output.out == output.err == ""
