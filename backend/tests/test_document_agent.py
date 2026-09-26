"""Unit tests for safe, non-diagnostic document extraction."""

from io import BytesIO

import pymupdf as fitz
from PIL import Image, ImageFilter
import pytest

from agents.document_agent import DocumentExtractionError, MAX_UPLOAD_BYTES, extract_document


def _text_pdf(text: str) -> bytes:
    """Create a small selectable-text PDF without committing patient-like fixtures."""
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


def _png() -> bytes:
    """Create a valid image for file-signature validation tests."""
    image = Image.new("RGB", (10, 10), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_text_pdf_extracts_cbc_values_without_interpretation():
    report = "Hemoglobin 11.2 g/dL 12-16\nWBC 6.4 x10^9/L 4.0-11.0"

    response = extract_document("cbc.pdf", _text_pdf(report))

    assert response.extraction_method == "pdf_text"
    assert response.report_type == "cbc"
    assert [item.model_dump() for item in response.results] == [
        {
            "test": "Hemoglobin",
            "value": 11.2,
            "unit": "g/dL",
            "reference_range": "12-16",
            "confidence": 0.99,
            "needs_verification": False,
        },
        {
            "test": "WBC",
            "value": 6.4,
            "unit": "x10^9/L",
            "reference_range": "4.0-11.0",
            "confidence": 0.99,
            "needs_verification": False,
        },
    ]


def test_blurry_scanned_image_is_flagged_with_low_confidence(monkeypatch):
    image = Image.new("L", (100, 40), color=190).filter(ImageFilter.GaussianBlur(radius=4))
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    monkeypatch.setattr(
        "agents.document_agent._ocr_image",
        lambda _image: ("Hemoglobin 11.2 g/dL 12-16", 0.35),
    )
    response = extract_document("blurry-cbc.png", buffer.getvalue())

    result = response.results[0]
    assert response.extraction_method == "ocr"
    assert response.report_type == "cbc"
    assert result.confidence < 0.75
    assert result.needs_verification is True


def test_unrecognized_test_is_returned_and_marked_for_verification():
    report = "Hemoglobin 11.2 g/dL 12-16\nNovel Marker 5.2 mg/dL 1.0-7.0"

    response = extract_document("cbc.pdf", _text_pdf(report))

    unknown = response.results[1]
    assert unknown.test == "Novel Marker"
    assert unknown.value == 5.2
    assert unknown.needs_verification is True
    assert unknown.confidence < 0.75


def test_lipid_pdf_is_classified_and_extracted():
    report = "LDL Cholesterol 110 mg/dL 0-129\nTriglycerides 98 mg/dL 0-149"

    response = extract_document("lipid.pdf", _text_pdf(report))

    assert response.report_type == "lipid"
    assert [result.test for result in response.results] == ["LDL Cholesterol", "Triglycerides"]


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("notes.txt", b"not a report", "Only PDF"),
        ("report.pdf", b"not a PDF", "does not contain valid PDF data"),
        ("report.png", b"not an image", "uploaded image file is invalid"),
        ("report.jpg", _png(), "extension does not match"),
    ],
)
def test_invalid_uploads_are_rejected_before_extraction(filename, content, message):
    with pytest.raises(DocumentExtractionError, match=message):
        extract_document(filename, content)


def test_oversized_upload_is_rejected_before_parsing():
    with pytest.raises(DocumentExtractionError, match="10 MB"):
        extract_document("report.pdf", b"x" * (MAX_UPLOAD_BYTES + 1))


def test_non_cbc_or_lipid_document_is_rejected():
    with pytest.raises(DocumentExtractionError, match="No supported CBC or Lipid"):
        extract_document("other.pdf", _text_pdf("Blood Glucose 95 mg/dL 70-100"))


def test_extract_endpoint_accepts_a_valid_pdf(client):
    response = client.post(
        "/agents/document/extract",
        files={"file": ("cbc.pdf", _text_pdf("Hemoglobin 11.2 g/dL 12-16"), "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["report_type"] == "cbc"
