"""Extract structured CBC and lipid-profile fields from uploaded lab reports.

This module only reads report content. It deliberately does not classify results as
normal or abnormal, diagnose, or alter values that may have been read incorrectly.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
import re
from typing import Annotated, Literal

import pymupdf as fitz
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
import pytesseract
from pytesseract import Output
import spacy

router = APIRouter(prefix="/agents/document", tags=["document-agent"])

SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
LOW_CONFIDENCE_THRESHOLD = 0.75

# Aliases are intentionally a limited CBC/lipid vocabulary. An unmatched row is
# still returned, flagged for verification, instead of being guessed or discarded.
TEST_ALIASES = {
    "Hemoglobin": ("hemoglobin", "haemoglobin", "hgb", "hb"),
    "WBC": ("wbc", "white blood cell", "white blood cells", "leukocyte"),
    "RBC": ("rbc", "red blood cell", "red blood cells", "erythrocyte"),
    "Platelets": ("platelet", "platelets", "plt"),
    "Hematocrit": ("hematocrit", "haematocrit", "hct", "pcv"),
    "MCV": ("mcv", "mean corpuscular volume"),
    "MCH": ("mch", "mean corpuscular hemoglobin"),
    "MCHC": ("mchc", "mean corpuscular hemoglobin concentration"),
    "RDW": ("rdw", "red cell distribution width"),
    "Total Cholesterol": ("total cholesterol", "cholesterol"),
    "HDL Cholesterol": ("hdl cholesterol", "hdl"),
    "LDL Cholesterol": ("ldl cholesterol", "ldl"),
    "Triglycerides": ("triglyceride", "triglycerides", "tg"),
    "VLDL Cholesterol": ("vldl cholesterol", "vldl"),
}
CBC_TESTS = {"Hemoglobin", "WBC", "RBC", "Platelets", "Hematocrit", "MCV", "MCH", "MCHC", "RDW"}
LIPID_TESTS = {"Total Cholesterol", "HDL Cholesterol", "LDL Cholesterol", "Triglycerides", "VLDL Cholesterol"}

NUMBER_RE = re.compile(r"[<>]?\s*(\d+(?:\.\d+)?)")
RANGE_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?\s*(?:-|to)\s*\d+(?:\.\d+)?)(?!\d)", re.IGNORECASE)
UNIT_RE = re.compile(
    r"^\s*("
    r"(?:x?\s*10\s*\^?\s*-?\d+\s*/\s*[A-Za-z%]+)"
    r"|(?:[A-Za-z%]+(?:\s*\^?\s*-?\d+)?(?:\s*/\s*[A-Za-z%]+)?)"
    r")"
)
HEADER_LABELS = {"test", "test name", "result", "value", "unit", "reference range", "normal range"}


class ExtractedLabResult(BaseModel):
    """One value transcribed from a lab report, without clinical interpretation."""

    test: str
    value: float
    unit: str | None = None
    reference_range: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    needs_verification: bool


class DocumentExtractionResponse(BaseModel):
    """Structured output and provenance for a report extraction request."""

    extraction_method: str
    report_type: Literal["cbc", "lipid"]
    results: list[ExtractedLabResult]


class DocumentExtractionError(ValueError):
    """Raised when an upload cannot be safely read as a supported document."""


@lru_cache(maxsize=1)
def _load_nlp():
    """Load the required spaCy NER model once per process.

    The blank fallback keeps the upload endpoint operational when the model has not
    yet been downloaded, while the deterministic CBC/lipid alias list marks fields
    appropriately. Setup instructions in the README install ``en_core_web_sm``.
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return spacy.blank("en")


def _canonical_test_name(label: str, named_entities: set[str]) -> tuple[str, bool]:
    """Return the known canonical test name, or the untouched report label."""
    normalised = re.sub(r"\s+", " ", label.strip()).lower()
    # Match specific aliases first: "LDL Cholesterol" must not be reduced to
    # the broader "cholesterol" alias for Total Cholesterol.
    aliases = sorted(
        ((alias, canonical) for canonical, names in TEST_ALIASES.items() for alias in names),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for alias, canonical in aliases:
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalised):
            return canonical, True

    # NER is run for every report. Preserve any entity-derived label as unverified
    # rather than assuming it maps to a supported test.
    if normalised in named_entities:
        return label.strip(), False
    return label.strip(), False


def _ner_terms(text: str) -> set[str]:
    """Run spaCy NER and return normalized entity text for extraction evidence."""
    document = _load_nlp()(text)
    return {re.sub(r"\s+", " ", entity.text.strip()).lower() for entity in document.ents}


def _parse_report_line(line: str, named_entities: set[str], source_confidence: float) -> ExtractedLabResult | None:
    """Parse a single report row containing a label, numeric result, unit and range."""
    number = NUMBER_RE.search(line)
    if number is None:
        return None

    label = line[: number.start()].strip(" :|\t-")
    if not label or label.lower() in HEADER_LABELS or not re.search(r"[A-Za-z]", label):
        return None

    # Table separators and OCR artefacts frequently appear after the label.
    label = re.sub(r"\s+", " ", label).strip()
    value = float(number.group(1))
    tail = line[number.end() :]
    range_match = RANGE_RE.search(tail)
    reference_range = range_match.group(1) if range_match else None
    unit_text = tail[: range_match.start()] if range_match else tail
    unit_match = UNIT_RE.search(unit_text)
    unit = re.sub(r"\s+", "", unit_match.group(1)) if unit_match else None

    test, is_known_test = _canonical_test_name(label, named_entities)
    confidence = source_confidence
    # A label outside the limited CBC/lipid vocabulary must be reviewed even if
    # its surrounding characters were read clearly.
    confidence += 0.20 if is_known_test else -0.35
    confidence += 0.08  # a numeric value was found in a label/value row
    confidence += 0.05 if unit else 0.0
    confidence += 0.10 if reference_range else 0.0
    confidence = round(min(confidence, 0.99), 2)
    needs_verification = (
        confidence < LOW_CONFIDENCE_THRESHOLD
        or not is_known_test
        or unit is None
        or reference_range is None
    )
    return ExtractedLabResult(
        test=test,
        value=value,
        unit=unit,
        reference_range=reference_range,
        confidence=confidence,
        needs_verification=needs_verification,
    )


def parse_lab_results(text: str, source_confidence: float) -> list[ExtractedLabResult]:
    """Extract unique rows from report text while retaining unknown test labels."""
    entities = _ner_terms(text)
    results: list[ExtractedLabResult] = []
    seen: set[tuple[str, float]] = set()
    for line in text.splitlines():
        result = _parse_report_line(line, entities, source_confidence)
        if result is None:
            continue
        key = (result.test.lower(), result.value)
        if key not in seen:
            seen.add(key)
            results.append(result)
    return results


def _ocr_image(image: Image.Image) -> tuple[str, float]:
    """Extract text and normalized word-confidence from an image using Tesseract."""
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    words = data.get("text", [])
    confidences = data.get("conf", [])
    # Keep Tesseract's line groups intact. Flattening all words into one string
    # would turn a multi-row report table into a single, incorrectly parsed row.
    lines: dict[tuple[object, object, object], list[str]] = {}
    for index, word in enumerate(words):
        if not str(word).strip():
            continue
        line_key = tuple(
            data.get(key, [0] * len(words))[index]
            for key in ("block_num", "par_num", "line_num")
        )
        lines.setdefault(line_key, []).append(str(word))
    text = "\n".join(" ".join(line_words) for line_words in lines.values())
    usable_confidences = [float(value) for value in confidences if str(value) not in {"", "-1"}]
    average = sum(usable_confidences) / len(usable_confidences) if usable_confidences else 0.0
    return text, max(0.0, min(1.0, average / 100))


def _extract_pdf_text(content: bytes) -> tuple[str, str, float]:
    """Prefer a selectable PDF text layer, rendering pages for OCR only if needed."""
    try:
        pdf = fitz.open(stream=content, filetype="pdf")
    except (fitz.FileDataError, RuntimeError) as exc:
        raise DocumentExtractionError("The PDF could not be read.") from exc

    try:
        text = "\n".join(page.get_text("text") for page in pdf).strip()
        if re.search(r"[A-Za-z]{2,}", text):
            return text, "pdf_text", 0.77

        ocr_text: list[str] = []
        page_scores: list[float] = []
        for page in pdf:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(BytesIO(pixmap.tobytes("png")))
            page_text, page_score = _ocr_image(image)
            ocr_text.append(page_text)
            page_scores.append(page_score)
        score = sum(page_scores) / len(page_scores) if page_scores else 0.0
        # OCR engines can report high character confidence while confusing a digit
        # in a medical value or range. Keep OCR-derived fields below the automatic
        # verification threshold; a human can confirm the report image.
        return "\n".join(ocr_text), "ocr", score * 0.25
    finally:
        pdf.close()


def _extract_image_text(content: bytes) -> tuple[str, str, float]:
    """Read a PNG/JPEG report with Tesseract OCR."""
    try:
        image = Image.open(BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise DocumentExtractionError("The image could not be read.") from exc
    text, score = _ocr_image(image)
    # See the equivalent PDF OCR path: never mark an OCR transcription as safe to
    # use without review, even when Tesseract's character score is high.
    return text, "ocr", score * 0.25


def _validate_upload(filename: str, content: bytes) -> str:
    """Validate the file extension, size, and file signature before extraction."""
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentExtractionError("Only PDF, PNG, JPG, and JPEG lab reports are supported.")
    if not content:
        raise DocumentExtractionError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise DocumentExtractionError("The uploaded file exceeds the 10 MB size limit.")
    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise DocumentExtractionError("The uploaded .pdf file does not contain valid PDF data.")
        return suffix

    try:
        image = Image.open(BytesIO(content))
        image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise DocumentExtractionError("The uploaded image file is invalid.") from exc

    expected_format = "PNG" if suffix == ".png" else "JPEG"
    if image.format != expected_format:
        raise DocumentExtractionError("The file extension does not match the uploaded image data.")
    return suffix


def _report_type(results: list[ExtractedLabResult]) -> Literal["cbc", "lipid"]:
    """Identify the supported report family from recognized, non-interpreted labels."""
    # A correctly identified test can establish the report family even when its
    # OCR-derived value needs human verification.
    recognized_tests = {result.test for result in results}
    is_cbc = bool(recognized_tests & CBC_TESTS)
    is_lipid = bool(recognized_tests & LIPID_TESTS)
    if is_cbc and not is_lipid:
        return "cbc"
    if is_lipid and not is_cbc:
        return "lipid"
    if is_cbc and is_lipid:
        raise DocumentExtractionError("Upload one CBC report or one Lipid Profile report, not a combined report.")
    raise DocumentExtractionError("No supported CBC or Lipid Profile tests were detected in this report.")


def extract_document(filename: str, content: bytes) -> DocumentExtractionResponse:
    """Extract CBC/lipid rows from uploaded PDF, PNG, or JPEG bytes."""
    suffix = _validate_upload(filename, content)

    if suffix == ".pdf":
        text, method, source_confidence = _extract_pdf_text(content)
    else:
        text, method, source_confidence = _extract_image_text(content)
    results = parse_lab_results(text, source_confidence)
    return DocumentExtractionResponse(
        extraction_method=method,
        report_type=_report_type(results),
        results=results,
    )


@router.post("/extract", response_model=DocumentExtractionResponse)
async def extract_report(file: Annotated[UploadFile, File(description="CBC or lipid report")]):
    """Upload a PDF/PNG/JPEG report and receive non-diagnostic structured fields."""
    try:
        content = await file.read()
        return extract_document(file.filename or "", content)
    except DocumentExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except pytesseract.TesseractNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tesseract OCR is not installed or available on this server.",
        ) from exc
