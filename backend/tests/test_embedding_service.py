"""Offline embedding tests: all model creation and inference are faked."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from agents import embedding_service as module
from agents.embedding_service import EmbeddingService, build_embedding_text
from agents.knowledge_base import KnowledgeBaseLoader


ORIGINAL_FACTORY = module._create_model


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    monkeypatch.setattr(module, "load_dotenv", lambda *args: None)
    for key in ("EMBEDDING_MODEL_NAME", "EMBEDDING_DEVICE", "EMBEDDING_LOCAL_FILES_ONLY", "EMBEDDING_MODEL_REVISION"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(module, "_create_model", Mock(side_effect=AssertionError("Real model loading prohibited")))


@pytest.fixture
def document():
    return KnowledgeBaseLoader().get_by_test_name("Hemoglobin")


def test_exact_hemoglobin_text(document):
    assert build_embedding_text(document) == (
        "Test: Hemoglobin\nAliases: Hb; Hgb; Haemoglobin\nTitle: Hemoglobin Test\n"
        "Definition: Hemoglobin is an iron-containing protein found in red blood cells that helps carry oxygen from the lungs to the body's tissues.\n"
        "What it measures: A hemoglobin test measures the amount of hemoglobin in a blood sample.\n"
        "General information: Hemoglobin testing is commonly included as part of a complete blood count. LabLens uses this information only for educational explanation and does not use it to diagnose a condition."
    )


def test_whitespace_metadata_exclusion_and_no_mutation(document):
    document.test_name = "  A\tTest "
    document.aliases = [" A-B ", " C\n D "]
    document.title = " Title\tCase "
    document.definition = " Definition:\n Keep-case! "
    document.what_it_measures = " A   measure. "
    document.general_information = " General\r\n information. "
    document.id = "EXCLUDED_ID"
    document.report_type = "EXCLUDED_REPORT"
    for name in ("publisher", "title", "url", "accessed_date"):
        setattr(document.source, name, "EXCLUDED_SOURCE_" + name)
    before = document.model_dump()
    expected = "Test: A Test\nAliases: A-B; C D\nTitle: Title Case\nDefinition: Definition: Keep-case!\nWhat it measures: A measure.\nGeneral information: General information."
    assert build_embedding_text(document) == expected
    assert build_embedding_text(document) == expected
    assert document.model_dump() == before


def test_lazy_loading_reuse_and_defaults():
    model = Mock()
    model.encode.return_value = [1, 0]
    factory = Mock(return_value=model)
    service = EmbeddingService(model_factory=factory)
    factory.assert_not_called()
    assert service.encode_text("first") == [1.0, 0.0]
    assert service.encode_text("second") == [1.0, 0.0]
    factory.assert_called_once_with(module.DEFAULT_MODEL_NAME, device="cpu", local_files_only=False)
    assert model.encode.call_count == 2
    model.encode.assert_called_with("second", normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def test_environment_settings_passed_to_factory(monkeypatch):
    for key, value in {"EMBEDDING_MODEL_NAME": "configured", "EMBEDDING_DEVICE": "cpu",
                       "EMBEDDING_LOCAL_FILES_ONLY": " TRUE ", "EMBEDDING_MODEL_REVISION": " abc "}.items():
        monkeypatch.setenv(key, value)
    model = Mock()
    model.encode.return_value = [1.0]
    factory = Mock(return_value=model)
    EmbeddingService(model_factory=factory).encode_text("text")
    factory.assert_called_once_with("configured", device="cpu", local_files_only=True, revision="abc")


def test_explicit_configuration_overrides_environment(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "environment")
    monkeypatch.setenv("EMBEDDING_LOCAL_FILES_ONLY", "invalid")
    model = Mock()
    model.encode.return_value = [1.0]
    factory = Mock(return_value=model)
    EmbeddingService(model_factory=factory, model_name="explicit", device="cpu",
                     local_files_only=True, revision="commit").encode_text("text")
    factory.assert_called_once_with("explicit", device="cpu", local_files_only=True, revision="commit")


def test_invalid_boolean_configuration(monkeypatch):
    monkeypatch.setenv("EMBEDDING_LOCAL_FILES_ONLY", "maybe")
    with pytest.raises(ValueError, match="true or false"):
        EmbeddingService()


def test_conflicting_injection_rejected():
    with pytest.raises(ValueError):
        EmbeddingService(model=Mock(), model_factory=Mock())


def test_failed_load_can_retry():
    model = Mock()
    model.encode.return_value = [1.0]
    factory = Mock(side_effect=[OSError("unavailable"), model])
    service = EmbeddingService(model_factory=factory)
    with pytest.raises(RuntimeError, match="Could not load embedding model") as exc:
        service.encode_text("text")
    assert isinstance(exc.value.__cause__, OSError)
    assert service.encode_text("text") == [1.0]
    assert factory.call_count == 2


@pytest.mark.parametrize("output", [[1, 2.5], (1, 2.5), np.array([1, 2.5], dtype=np.float32)])
def test_single_output_conversion(output):
    model = Mock()
    model.encode.return_value = output
    result = EmbeddingService(model=model).encode_text("Text")
    assert result == [1.0, 2.5]
    assert all(type(value) is float for value in result)


@pytest.mark.parametrize("text", ["", " ", "\t\n"])
def test_blank_input_does_not_load(text):
    factory = Mock()
    with pytest.raises(ValueError, match="blank"):
        EmbeddingService(model_factory=factory).encode_text(text)
    factory.assert_not_called()


@pytest.mark.parametrize("output", [None, [], 1.0, "1", ["1"], [True], [[1.0]],
                                        [float("nan")], [float("inf")], [float("-inf")], [1j], np.empty((0,))])
def test_bad_single_vector_rejected(output):
    model = Mock()
    model.encode.return_value = output
    with pytest.raises(ValueError):
        EmbeddingService(model=model).encode_text("text")


def test_empty_batch_does_not_load():
    factory = Mock()
    assert EmbeddingService(model_factory=factory).encode_documents([]) == []
    factory.assert_not_called()


def test_batch_order_one_call_and_normalization(document):
    other = document.model_copy(deep=True)
    other.test_name = "Second"
    model = Mock()
    model.encode.return_value = np.array([[1, 0], [0, 1]])
    assert EmbeddingService(model=model).encode_documents([other, document]) == [[1.0, 0.0], [0.0, 1.0]]
    model.encode.assert_called_once_with([build_embedding_text(other), build_embedding_text(document)],
                                        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


@pytest.mark.parametrize("output", [None, [], [[1]], [[1], [1, 2]], [[], []],
                                        [[1], [float("nan")]], [[1], [float("inf")]],
                                        [[1], ["2"]], [1, 2], [[[1]], [[2]]]])
def test_bad_batch_rejected(document, output):
    model = Mock()
    model.encode.return_value = output
    with pytest.raises(ValueError):
        EmbeddingService(model=model).encode_documents([document, document])


def test_encoding_failure_is_not_replaced_with_vectors():
    model = Mock()
    model.encode.side_effect = RuntimeError("encoding failed")
    with pytest.raises(RuntimeError, match="encoding failed"):
        EmbeddingService(model=model).encode_text("text")


def test_default_factory_uses_sentence_transformer_lazily(monkeypatch):
    # Substitute the module itself: no real SentenceTransformer is imported/created.
    model = Mock()
    model.encode.return_value = [1.0]
    constructor = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))
    # Use the original factory with the fake module, bypassing the autouse guard.
    service = EmbeddingService(model_factory=ORIGINAL_FACTORY, local_files_only=True)
    constructor.assert_not_called()
    service.encode_text("text")
    constructor.assert_called_once_with(module.DEFAULT_MODEL_NAME, device="cpu", local_files_only=True)
