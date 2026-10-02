"""Validation and JSON serialization of Retrieval Agent contracts."""

import json

from pydantic import ValidationError
import pytest

from agents.retrieval_models import (
    RetrievalMatch,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
    RetrievalSource,
)


IDS = {"task_id": "task-1", "report_id": "report-1", "user_id": "user-1"}
SOURCE = {
    "title": "Hemoglobin Test",
    "url": "https://medlineplus.gov/lab-tests/hemoglobin-test/",
}
MATCH = {"information": "Hemoglobin carries oxygen.", "sources": [SOURCE]}


@pytest.mark.parametrize("names", [["Hemoglobin"], ["HDL", "LDL", "Novel Marker"]])
def test_request_accepts_one_or_more_names_and_preserves_ids(names):
    request = RetrievalRequest(**IDS, test_names=names)
    assert request.model_dump() == {**IDS, "test_names": names}


@pytest.mark.parametrize("names", [[], [""], [" \t\n"], ["WBC", " "], [None]])
def test_request_rejects_empty_or_invalid_test_names(names):
    with pytest.raises(ValidationError):
        RetrievalRequest(**IDS, test_names=names)


def test_test_names_trim_surrounding_whitespace():
    assert RetrievalRequest(**IDS, test_names=[" WBC "]).test_names == ["WBC"]
    assert RetrievalResult(test_name=" LDL ", found=False).test_name == "LDL"


def test_response_preserves_ids_results_and_source_attribution_in_json():
    payload = {
        **IDS,
        "results": [
            {"test_name": "Hemoglobin", "found": True, "matches": [MATCH]},
            {"test_name": "Novel Marker", "found": False, "matches": []},
        ],
    }
    response = RetrievalResponse.model_validate(payload)
    assert json.loads(response.model_dump_json()) == payload
    assert RetrievalResponse.model_validate_json(response.model_dump_json()) == response


def test_unsupported_test_has_no_matches():
    result = RetrievalResult(test_name="Novel Marker", found=False)
    assert result.model_dump() == {
        "test_name": "Novel Marker", "found": False, "matches": [],
    }


@pytest.mark.parametrize("found,matches", [(True, []), (False, [MATCH])])
def test_found_must_agree_with_evidence(found, matches):
    with pytest.raises(ValidationError):
        RetrievalResult(test_name="Hemoglobin", found=found, matches=matches)


@pytest.mark.parametrize("name", ["", " \n"])
def test_result_rejects_blank_test_name(name):
    with pytest.raises(ValidationError):
        RetrievalResult(test_name=name, found=False)


@pytest.mark.parametrize("sources", [[], [{}], [{"title": "Source"}]])
def test_match_requires_source_attribution(sources):
    with pytest.raises(ValidationError):
        RetrievalMatch(information="Educational text.", sources=sources)


def test_match_rejects_missing_sources():
    with pytest.raises(ValidationError):
        RetrievalMatch(information="Educational text.")


@pytest.mark.parametrize("information", ["", " \t"])
def test_match_rejects_blank_information(information):
    with pytest.raises(ValidationError):
        RetrievalMatch(information=information, sources=[SOURCE])


@pytest.mark.parametrize("changes", [
    {"title": " "}, {"url": ""}, {"url": "not-a-url"},
    {"url": "javascript:alert(1)"},
])
def test_source_requires_title_and_http_url(changes):
    with pytest.raises(ValidationError):
        RetrievalSource(**{**SOURCE, **changes})


def test_response_requires_at_least_one_result():
    with pytest.raises(ValidationError):
        RetrievalResponse(**IDS, results=[])


@pytest.mark.parametrize("missing", list(IDS))
@pytest.mark.parametrize("model,body", [
    (RetrievalRequest, {"test_names": ["WBC"]}),
    (RetrievalResponse, {"results": [{"test_name": "WBC", "found": False}]}),
])
def test_correlation_ids_are_required(model, body, missing):
    ids = {key: value for key, value in IDS.items() if key != missing}
    with pytest.raises(ValidationError):
        model(**ids, **body)
