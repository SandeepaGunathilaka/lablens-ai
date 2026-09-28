"""Offline knowledge-base validation using synthetic, non-medical fixtures."""

import json

import pytest

from agents.knowledge_base import KnowledgeBaseValidationError, validate_knowledge_base


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
            "url": "https://example.invalid/source", "accessed_date": "2026-09-28",
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
