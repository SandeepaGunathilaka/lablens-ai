"""Deterministic retrieval, collision safety, and preservation of KB evidence."""

import json
from unittest.mock import Mock

import pytest

from agents.knowledge_base import KnowledgeBaseLoader, KnowledgeBaseValidationError
from agents.keyword_retriever import KeywordRetriever, normalize_test_name


NAMES = ["Hemoglobin", "WBC", "Platelets", "HDL", "LDL", "Triglycerides", "Total Cholesterol"]
ALIASES = [
    ("Hb", "Hemoglobin"), ("Hgb", "Hemoglobin"), ("Haemoglobin", "Hemoglobin"),
    ("White Blood Cell", "WBC"), ("White Blood Cells", "WBC"),
    ("White Blood Cell Count", "WBC"), ("White Cell Count", "WBC"), ("Leukocyte Count", "WBC"),
    ("Platelet", "Platelets"), ("Platelet Count", "Platelets"),
    ("PLT", "Platelets"), ("Thrombocyte Count", "Platelets"),
    ("HDL Cholesterol", "HDL"), ("HDL-C", "HDL"),
    ("High-Density Lipoprotein Cholesterol", "HDL"), ("High-Density Lipoprotein", "HDL"),
    ("LDL Cholesterol", "LDL"), ("LDL-C", "LDL"),
    ("Low-Density Lipoprotein Cholesterol", "LDL"), ("Low-Density Lipoprotein", "LDL"),
    ("Triglyceride", "Triglycerides"), ("TG", "Triglycerides"), ("TRIG", "Triglycerides"),
    ("Cholesterol Total", "Total Cholesterol"), ("Total Chol", "Total Cholesterol"),
    ("TC", "Total Cholesterol"),
]


@pytest.mark.parametrize("name", NAMES)
def test_canonical_names(name):
    assert KeywordRetriever().retrieve(name).test_name == name


@pytest.mark.parametrize("alias,expected", ALIASES)
def test_approved_aliases(alias, expected):
    assert KeywordRetriever().retrieve(alias).test_name == expected


def test_alias_cases_cover_current_kb():
    actual = {(alias, doc.test_name) for doc in KnowledgeBaseLoader().load_all() for alias in doc.aliases}
    assert actual == set(ALIASES)


@pytest.mark.parametrize("query,expected", [
    ("hemoglobin", "Hemoglobin"), ("  HEMOGLOBIN  ", "Hemoglobin"),
    (" Hb ", "Hemoglobin"), ("White \t Blood   Cell\nCount", "WBC"),
    ("HDL \u2013 C", "HDL"), ("HDL\u2014C", "HDL"),
    ("HDL\u2011C", "HDL"), ("High Density Lipoprotein", "HDL"),
    ("Low\u2010Density Lipoprotein", "LDL"),
])
def test_normalized_matching(query, expected):
    assert KeywordRetriever().retrieve(query).test_name == expected


@pytest.mark.parametrize("query", [
    "Cholesterol", "Blood Cell Count", "HDL Cholesterol result", "Hemoglob",
    "completely unknown test", "", " \t\n", "HDLC", "HDL/C", "HDL.C",
    "Count Cell Blood White", "White Blood Cell Counts",
])
def test_no_guessing(query):
    assert KeywordRetriever().retrieve(query) is None


def test_casefold_and_punctuation_preservation():
    assert normalize_test_name(" Straße / HDL-C ") == "strasse / hdl c"


def write_doc(directory, identifier, name, aliases):
    payload = {
        "id": identifier, "test_name": name, "aliases": aliases, "report_type": "CBC",
        "title": "Fixture", "definition": "Fixture definition.",
        "what_it_measures": "Fixture measurement.", "general_information": "Fixture context.",
        "source": {"publisher": "Fixture", "title": "Fixture source",
                   "url": "https://example.invalid/source", "accessed_date": "2026-09-28"},
    }
    (directory / f"{identifier}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("name_a,aliases_a,name_b,aliases_b,key", [
    ("First", ["Shared"], "Second", ["shared"], "shared"),
    ("First", ["Other"], "Second", ["First"], "first"),
    ("First", ["Shared-Term"], "Second", ["Shared\u2011Term"], "shared term"),
])
def test_collisions_reject_entire_index_and_retry(
    tmp_path, name_a, aliases_a, name_b, aliases_b, key,
):
    write_doc(tmp_path, "a", name_a, aliases_a)
    write_doc(tmp_path, "b", name_b, aliases_b)
    retriever = KeywordRetriever(KnowledgeBaseLoader(tmp_path))
    for query in [name_a, name_b]:
        with pytest.raises(KnowledgeBaseValidationError) as exc:
            retriever.retrieve(query)
        assert key in str(exc.value)
        assert name_a in str(exc.value)
        assert name_b in str(exc.value)
    write_doc(tmp_path, "b", name_b, ["Distinct"])
    assert retriever.retrieve("Distinct").test_name == name_b
    assert retriever.retrieve(name_a).test_name == name_a


def test_same_document_duplicates_are_harmless(tmp_path):
    write_doc(tmp_path, "a", "Sample-Test", ["sample test", " SAMPLE\u2013TEST "])
    assert KeywordRetriever(KnowledgeBaseLoader(tmp_path)).retrieve("sample test").id == "a"


def test_lazy_loading_and_index_reuse():
    loader = KnowledgeBaseLoader()
    loader.load_all = Mock(wraps=loader.load_all)
    retriever = KeywordRetriever(loader)
    loader.load_all.assert_not_called()
    assert retriever.retrieve(" ") is None
    loader.load_all.assert_not_called()
    assert retriever.retrieve("Hb").test_name == "Hemoglobin"
    assert retriever.retrieve("WBC").test_name == "WBC"
    assert retriever.retrieve("Unknown") is None
    loader.load_all.assert_called_once_with()


def test_validation_failure_propagates_and_retries(tmp_path):
    retriever = KeywordRetriever(KnowledgeBaseLoader(tmp_path))
    with pytest.raises(KnowledgeBaseValidationError):
        retriever.retrieve("Sample")
    write_doc(tmp_path, "a", "Sample", ["Example"])
    assert retriever.retrieve("Example").test_name == "Sample"


def test_deep_copy_and_complete_metadata_preservation():
    documents = KnowledgeBaseLoader().load_all()
    originals = [doc.model_dump() for doc in documents]
    loader = Mock(spec=KnowledgeBaseLoader)
    loader.load_all.return_value = documents
    retriever = KeywordRetriever(loader)
    for original in originals:
        result = retriever.retrieve(original["test_name"])
        assert result.model_dump() == original
        result.source.publisher = "Changed"
        result.source.title = "Changed"
        result.source.url = "Changed"
        result.source.accessed_date = "Changed"
        result.aliases.clear()
        result.definition = "Changed"
        result.test_name = "Changed"
        assert retriever.retrieve(original["test_name"]).model_dump() == original
    assert [doc.model_dump() for doc in documents] == originals
