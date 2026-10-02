"""Offline ranked-search tests: no model, network, or persistent client."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents import semantic_retriever as module
from agents.knowledge_base import KnowledgeDocument
from agents.knowledge_base_fingerprint import document_sha256
from agents.semantic_retriever import SemanticRetriever, SemanticRetrievalIntegrityError, SemanticSearchResult
from agents.vector_store import VectorSearchResult, VectorIndexStaleError, VectorIndexNotBuiltError


@pytest.fixture
def setup():
    docs = [KnowledgeDocument(id=str(i), test_name=f"Test {i}", aliases=[f"Alias {i}"],
        report_type="CBC", title="Educational fixture", definition="Synthetic definition",
        what_it_measures="Synthetic measurement", general_information="Synthetic context",
        source={"publisher":"Fixture", "title":"Fixture source", "url":"https://example.org", "accessed_date":"2026-09-29"})
        for i in range(3)]
    encoder = SimpleNamespace(model_name="fake", revision=None, encode_text=Mock(return_value=[1.0, 0.0]))
    candidates = [VectorSearchResult(doc.id, distance, {"document_sha256":document_sha256(doc)}, "not authoritative")
                  for doc, distance in zip(reversed(docs), [0.4, 0.1, 1.2])]
    store = SimpleNamespace(model_name="fake", model_revision=None, dimension=2,
                            kb_directory="unused", query=Mock(return_value=candidates))
    loader = SimpleNamespace(load_all=Mock(return_value=docs))
    return SemanticRetriever(encoder, store, loader), encoder, store, loader, docs


def test_ranked_search_preserves_data_and_encodes_once(setup):
    retriever, encoder, store, loader, docs = setup
    before = [doc.model_dump() for doc in docs]
    results = retriever.search("  Oxygen  Carrying-Protein? \n")
    encoder.encode_text.assert_called_once_with("Oxygen  Carrying-Protein?")
    store.query.assert_called_once_with(encoder.encode_text.return_value, n_results=3)
    loader.load_all.assert_called_once_with()
    assert [row.document_id for row in results] == ["2", "1", "0"]
    assert [row.rank for row in results] == [1, 2, 3]
    assert [row.distance for row in results] == [0.4, 0.1, 1.2]
    for result, doc in zip(results, reversed(docs)):
        assert result.similarity == 1 - result.distance
        assert result.test_name == doc.test_name
        assert result.document == doc
        assert result.document is not doc
        assert result.document.source.model_dump() == doc.source.model_dump()
    assert [doc.model_dump() for doc in docs] == before
    results[0].document.source.title = "caller edit"
    assert docs[2].source.title == "Fixture source"


@pytest.mark.parametrize("distance", [-0.01, 1.2, 2.0])
def test_properties_are_unclamped_and_frozen(setup, distance):
    result = SemanticSearchResult("0", setup[4][0], distance, 1)
    assert result.similarity == 1 - distance
    for field, value in [("rank", 2), ("similarity", 0.5), ("test_name", "changed")]:
        with pytest.raises(FrozenInstanceError):
            setattr(result, field, value)


@pytest.mark.parametrize("query", ["", " \t\n", None, 3, [], True])
def test_invalid_query_before_work(setup, query):
    retriever, encoder, store, loader, _ = setup
    with pytest.raises(ValueError, match="Query"):
        retriever.search(query)
    encoder.encode_text.assert_not_called()
    store.query.assert_not_called()
    loader.load_all.assert_not_called()


@pytest.mark.parametrize("count", [0, -1, 1.5, "3", True, None])
def test_invalid_count_before_work(setup, count):
    retriever, encoder, store, loader, _ = setup
    with pytest.raises(ValueError, match="n_results"):
        retriever.search("text", count)
    encoder.encode_text.assert_not_called()
    store.query.assert_not_called()
    loader.load_all.assert_not_called()


def test_empty_results(setup):
    retriever, _, store, loader, _ = setup
    store.query.return_value = []
    assert retriever.search("text", 7) == []
    assert store.query.call_args.kwargs == {"n_results": 7}
    loader.load_all.assert_not_called()


@pytest.mark.parametrize("failure", ["unknown", "missing_hash", "wrong_hash", "changed_source"])
def test_integrity_failure(setup, failure):
    retriever, _, store, _, docs = setup
    candidate = store.query.return_value[0]
    if failure == "unknown":
        store.query.return_value[0] = VectorSearchResult("unknown", 0.1, {}, "text")
    elif failure == "missing_hash":
        candidate.metadata.clear()
    elif failure == "wrong_hash":
        candidate.metadata["document_sha256"] = "wrong"
    else:
        docs[2].source.url = "https://example.org/changed"
    with pytest.raises(SemanticRetrievalIntegrityError):
        retriever.search("text")


@pytest.mark.parametrize("error", [VectorIndexStaleError("stale"), VectorIndexNotBuiltError("missing")])
def test_store_errors_propagate(setup, error):
    retriever, _, store, loader, _ = setup
    store.query.side_effect = error
    with pytest.raises(type(error)) as caught:
        retriever.search("text")
    assert caught.value is error
    loader.load_all.assert_not_called()


def test_encoding_error_propagates(setup):
    retriever, encoder, store, _, _ = setup
    error = RuntimeError("encoder unavailable")
    encoder.encode_text.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        retriever.search("text")
    assert caught.value is error
    store.query.assert_not_called()


@pytest.mark.parametrize("field", ["model_name", "revision"])
def test_incompatible_encoder_before_encoding(setup, field):
    retriever, encoder, store, _, _ = setup
    setattr(encoder, field, "different")
    with pytest.raises(VectorIndexStaleError, match="model/revision"):
        retriever.search("text")
    encoder.encode_text.assert_not_called()
    store.query.assert_not_called()


@pytest.mark.parametrize("vector", [[1.0], [0.0, 0.0], [2.0, 0.0], [float("nan"), 0.0]])
def test_vector_configuration_checked(setup, vector):
    retriever, encoder, store, _, _ = setup
    encoder.encode_text.return_value = vector
    with pytest.raises(ValueError):
        retriever.search("text")
    store.query.assert_not_called()


def test_no_writes_query_retention_or_logging(setup, caplog, capsys):
    retriever, _, store, _, _ = setup
    for name in ("rebuild", "delete_collection", "add", "upsert", "modify"):
        setattr(store, name, Mock(side_effect=AssertionError("Writes forbidden")))
    original_keys = set(vars(retriever))
    results = retriever.search("private synthetic question")
    assert set(vars(retriever)) == original_keys
    assert "private synthetic question" not in repr(results)
    for name in ("rebuild", "delete_collection", "add", "upsert", "modify"):
        getattr(store, name).assert_not_called()
    assert caplog.text == ""
    assert capsys.readouterr().out == ""


def test_default_dependencies_are_existing_components(monkeypatch, setup):
    _, encoder, store, loader, _ = setup
    factories = [Mock(return_value=component) for component in (store, encoder, loader)]
    for name, factory in zip(("ChromaVectorStore", "EmbeddingService", "KnowledgeBaseLoader"), factories):
        monkeypatch.setattr(module, name, factory)
    SemanticRetriever()
    factories[0].assert_called_once_with()
    factories[1].assert_called_once_with(model_name="fake", revision="")
    factories[2].assert_called_once_with("unused")
    encoder.encode_text.assert_not_called()
    store.query.assert_not_called()
    loader.load_all.assert_not_called()
