"""Offline knowledge-base validation using synthetic, non-medical fixtures."""

import json

import pytest

from agents import knowledge_base
from agents.knowledge_base import (
    KnowledgeBaseLoader,
    KnowledgeBaseValidationError,
    KnowledgeDocument,
    validate_knowledge_base,
)


FIELDS = [
    "id", "test_name", "aliases", "report_type", "title", "definition",
    "what_it_measures", "general_information", "source.publisher",
    "source.title", "source.url", "source.accessed_date",
]


@pytest.fixture
def document():
    return {
        "id": "sample", "test_name": "Sample", "aliases": ["Example"],
        "report_type": "CBC", "title": "Sample title", "definition": "Fixture text.",
        "what_it_measures": "Fixture text.", "general_information": "Fixture text.",
        "source": {
            "publisher": "Example", "title": "Example source",
            "url": "https://medlineplus.gov/source", "accessed_date": "2026-09-28",
        },
    }


def write_document(directory, document, filename="sample.json"):
    path = directory / filename
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_discovers_json_and_ignores_readme(tmp_path, document):
    write_document(tmp_path, document)
    write_document(tmp_path, {**document, "id": "second"}, "second.json")
    (tmp_path / "README.md").write_text("Not JSON", encoding="utf-8")
    assert [item.id for item in validate_knowledge_base(tmp_path)] == ["sample", "second"]


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("invalid", ["missing", "empty", "blank", "null"])
def test_rejects_missing_or_empty_required_fields(tmp_path, document, field, invalid):
    parts = field.split(".")
    target = document["source"] if len(parts) == 2 else document
    key = parts[-1]
    if invalid == "missing":
        del target[key]
    elif invalid == "null":
        target[key] = None
    elif field == "aliases":
        target[key] = [] if invalid == "empty" else [" \t"]
    else:
        target[key] = "" if invalid == "empty" else " \t\n"
    write_document(tmp_path, document)
    with pytest.raises(KnowledgeBaseValidationError) as exc:
        validate_knowledge_base(tmp_path)
    assert "sample.json" in str(exc.value)
    assert field in str(exc.value)


@pytest.mark.parametrize("second_id", ["sample", " sample "])
def test_rejects_duplicate_ids_and_names_both_files(tmp_path, document, second_id):
    write_document(tmp_path, document, "first.json")
    write_document(tmp_path, {**document, "id": second_id}, "second.json")
    with pytest.raises(KnowledgeBaseValidationError) as exc:
        validate_knowledge_base(tmp_path)
    assert "duplicate id" in str(exc.value)
    assert "first.json" in str(exc.value)
    assert "second.json" in str(exc.value)


def test_reports_all_bad_files_instead_of_stopping_at_first(tmp_path, document):
    (tmp_path / "broken.json").write_text('{"id":', encoding="utf-8")
    write_document(tmp_path, {**document, "title": ""}, "empty.json")
    with pytest.raises(KnowledgeBaseValidationError) as exc:
        validate_knowledge_base(tmp_path)
    assert "broken.json" in str(exc.value)
    assert "empty.json" in str(exc.value)


@pytest.mark.parametrize("payload", [[], None, "text", {"source": []}])
def test_rejects_invalid_document_structure(tmp_path, payload):
    write_document(tmp_path, payload)
    with pytest.raises(KnowledgeBaseValidationError, match="sample.json"):
        validate_knowledge_base(tmp_path)


def test_rejects_invalid_encoding(tmp_path):
    (tmp_path / "broken.json").write_bytes(b"\xff")
    with pytest.raises(KnowledgeBaseValidationError, match="broken.json"):
        validate_knowledge_base(tmp_path)


def test_rejects_empty_directory(tmp_path):
    with pytest.raises(KnowledgeBaseValidationError, match="no JSON"):
        validate_knowledge_base(tmp_path)


def test_rejects_missing_directory(tmp_path):
    with pytest.raises(KnowledgeBaseValidationError, match="does not exist"):
        validate_knowledge_base(tmp_path / "missing")


def test_default_directory_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    documents = validate_knowledge_base()
    assert len(documents) == 7
    assert {item.test_name for item in documents} == {
        "Hemoglobin", "WBC", "Platelets", "HDL", "LDL", "Triglycerides", "Total Cholesterol",
    }


def test_loader_returns_seven_current_documents():
    documents = KnowledgeBaseLoader().load_all()
    assert len(documents) == 7
    assert all(isinstance(document, KnowledgeDocument) for document in documents)
    assert {document.test_name for document in documents} == {
        "Hemoglobin", "WBC", "Platelets", "HDL", "LDL", "Triglycerides", "Total Cholesterol",
    }


def test_loader_is_lazy_and_validates_only_once(tmp_path, document, monkeypatch):
    path = write_document(tmp_path, document)
    calls = []

    def tracked_validation(directory):
        calls.append(directory)
        return validate_knowledge_base(directory)

    monkeypatch.setattr(knowledge_base, "validate_knowledge_base", tracked_validation)
    loader = KnowledgeBaseLoader(tmp_path)
    assert calls == []
    first = loader.load_all()
    path.write_text("invalid JSON", encoding="utf-8")
    assert loader.load_all() == first
    assert loader.get_by_test_name("Sample") == first[0]
    assert calls == [tmp_path]


@pytest.mark.parametrize("query,expected", [
    ("Hemoglobin", "Hemoglobin"),
    ("hemoglobin", "Hemoglobin"),
    ("  HEMOGLOBIN  ", "Hemoglobin"),
    ("WBC", "WBC"),
])
def test_loader_canonical_lookup_loads_on_demand(query, expected):
    result = KnowledgeBaseLoader().get_by_test_name(query)
    assert isinstance(result, KnowledgeDocument)
    assert result.test_name == expected


@pytest.mark.parametrize("query", ["Unknown Test", "Hb", "HDL-C", "", "   "])
def test_loader_does_not_match_unknown_names_or_aliases(query):
    assert KnowledgeBaseLoader().get_by_test_name(query) is None


def test_loader_retries_after_validation_failure_without_partial_cache(tmp_path, document):
    write_document(tmp_path, document, "first.json")
    second = {**document, "id": "second", "test_name": "Second", "title": ""}
    write_document(tmp_path, second, "second.json")
    loader = KnowledgeBaseLoader(tmp_path)
    with pytest.raises(KnowledgeBaseValidationError):
        loader.load_all()
    with pytest.raises(KnowledgeBaseValidationError):
        loader.get_by_test_name("Sample")
    second["title"] = "Corrected title"
    write_document(tmp_path, second, "second.json")
    assert len(loader.load_all()) == 2
    assert loader.get_by_test_name("Second").title == "Corrected title"


@pytest.mark.parametrize("first_name,second_name", [
    ("Sample", "  SAMPLE  "), ("Straße", "STRASSE"),
])
def test_loader_rejects_normalized_duplicate_names_and_can_retry(
    tmp_path, document, first_name, second_name,
):
    write_document(tmp_path, {**document, "test_name": first_name}, "first.json")
    second = {**document, "id": "second", "test_name": second_name}
    write_document(tmp_path, second, "second.json")
    loader = KnowledgeBaseLoader(tmp_path)
    with pytest.raises(KnowledgeBaseValidationError) as exc:
        loader.load_all()
    assert "duplicate canonical test name" in str(exc.value)
    assert "'sample'" in str(exc.value)  # First document ID, independent of name.
    assert "'second'" in str(exc.value)
    second["test_name"] = "Distinct"
    write_document(tmp_path, second, "second.json")
    assert len(loader.load_all()) == 2
    assert loader.get_by_test_name("Distinct").id == "second"


def test_loader_accepts_custom_string_directory(tmp_path, document):
    write_document(tmp_path, document)
    loader = KnowledgeBaseLoader(str(tmp_path))
    assert loader.directory == tmp_path
    assert [item.id for item in loader.load_all()] == ["sample"]
    assert loader.get_by_test_name(" sample ").id == "sample"


def test_loader_returns_deep_copies(tmp_path, document):
    write_document(tmp_path, document)
    loader = KnowledgeBaseLoader(tmp_path)
    documents = loader.load_all()
    documents[0].test_name = "Changed"
    documents[0].aliases.append("Changed")
    documents[0].source.publisher = "Changed"
    documents.clear()
    result = loader.get_by_test_name("Sample")
    assert result.model_dump() == document
    result.source.title = "Changed"
    result.aliases.clear()
    assert loader.load_all()[0].model_dump() == document
    assert loader.get_by_test_name("Sample").model_dump() == document
