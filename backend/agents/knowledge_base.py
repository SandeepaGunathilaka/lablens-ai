"""Offline structural validation of curated knowledge documents.

Run from backend with ``python -m agents.knowledge_base``. Validation checks
structure and required content, not medical accuracy or source availability.
"""

import json
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, ValidationError


KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"
RequiredText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class KnowledgeSource(BaseModel):
    publisher: RequiredText
    title: RequiredText
    url: RequiredText
    accessed_date: RequiredText


class KnowledgeDocument(BaseModel):
    id: RequiredText
    test_name: RequiredText
    aliases: list[RequiredText] = Field(min_length=1)
    report_type: RequiredText
    title: RequiredText
    definition: RequiredText
    what_it_measures: RequiredText
    general_information: RequiredText
    source: KnowledgeSource


class KnowledgeBaseValidationError(ValueError):
    """All discovered validation failures, labeled with their file paths."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Knowledge-base validation failed:\n" + "\n".join(errors))


def validate_knowledge_base(directory: Path | str = KNOWLEDGE_BASE_DIR) -> list[KnowledgeDocument]:
    """Validate every JSON file and return documents only if all files pass.

    A directory override supports isolated tests. No files are written and no
    sources are fetched. IDs are compared after trimming surrounding whitespace.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise KnowledgeBaseValidationError([f"{directory}: knowledge-base directory does not exist"])
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise KnowledgeBaseValidationError([f"{directory}: no JSON knowledge documents found"])

    errors: list[str] = []
    documents: list[KnowledgeDocument] = []
    seen_ids: dict[str, Path] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"{path}: could not parse/read JSON: {exc}")
            continue

        # Check IDs even when a different field in the same document is invalid.
        document_id = payload.get("id") if isinstance(payload, dict) else None
        if isinstance(document_id, str) and document_id.strip():
            document_id = document_id.strip()
            if document_id in seen_ids:
                errors.append(f"{path}: duplicate id {document_id!r}; first used in {seen_ids[document_id]}")
            else:
                seen_ids[document_id] = path
        try:
            documents.append(KnowledgeDocument.model_validate(payload))
        except ValidationError as exc:
            for error in exc.errors():
                field = ".".join(str(part) for part in error["loc"]) or "document"
                errors.append(f"{path}: {field}: {error['msg']}")

    if errors:
        raise KnowledgeBaseValidationError(errors)
    return documents


class KnowledgeBaseLoader:
    """Lazily load a validated snapshot with canonical-name lookup only.

    A successful snapshot is reused for this instance's lifetime. Returned models
    are deep copies so callers cannot mutate cached documents or their sources.
    """

    def __init__(self, directory: Path | str = KNOWLEDGE_BASE_DIR):
        self.directory = Path(directory)
        self._documents: list[KnowledgeDocument] | None = None
        self._by_test_name: dict[str, KnowledgeDocument] = {}

    def _ensure_loaded(self) -> None:
        if self._documents is not None:
            return

        documents = validate_knowledge_base(self.directory)
        index: dict[str, KnowledgeDocument] = {}
        errors: list[str] = []
        for document in documents:
            name = document.test_name.strip().casefold()
            if name in index:
                errors.append(
                    f"{self.directory}: duplicate canonical test name {name!r} "
                    f"in documents {index[name].id!r} and {document.id!r}"
                )
            else:
                index[name] = document
        if errors:
            raise KnowledgeBaseValidationError(errors)

        # Publish only after both document validation and indexing succeed.
        self._by_test_name = index
        self._documents = documents

    def load_all(self) -> list[KnowledgeDocument]:
        self._ensure_loaded()
        assert self._documents is not None
        return [document.model_copy(deep=True) for document in self._documents]

    def clear_cache(self) -> None:
        """Allow a consumer to retry after rejecting a loaded snapshot."""
        self._documents = None
        self._by_test_name = {}

    def get_by_test_name(self, test_name: str) -> KnowledgeDocument | None:
        self._ensure_loaded()
        document = self._by_test_name.get(test_name.strip().casefold())
        return document.model_copy(deep=True) if document is not None else None


def main() -> int:
    try:
        documents = validate_knowledge_base()
    except KnowledgeBaseValidationError as exc:
        print(exc)
        return 1
    print(f"Validated {len(documents)} knowledge documents in {KNOWLEDGE_BASE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
