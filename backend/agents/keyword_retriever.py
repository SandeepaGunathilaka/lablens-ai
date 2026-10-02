"""Exact canonical/approved-alias retrieval from the curated knowledge base."""

from agents.knowledge_base import (
    KnowledgeBaseLoader,
    KnowledgeBaseValidationError,
    KnowledgeDocument,
)


# ASCII hyphen, Unicode hyphen, non-breaking hyphen, figure/en/em dash.
_DASH_TRANSLATION = str.maketrans({char: " " for char in "-\u2010\u2011\u2012\u2013\u2014"})


def normalize_test_name(value: str) -> str:
    """Preserve word order and other punctuation; never guess missing words."""
    return " ".join(value.strip().casefold().translate(_DASH_TRANSLATION).split())


class KeywordRetriever:
    """Lazily index canonical names and aliases, returning isolated KB copies."""

    def __init__(self, loader: KnowledgeBaseLoader | None = None):
        self._loader = loader if loader is not None else KnowledgeBaseLoader()
        self._index: dict[str, KnowledgeDocument] | None = None

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        index: dict[str, KnowledgeDocument] = {}
        errors: list[str] = []
        for document in self._loader.load_all():
            for name in [document.test_name, *document.aliases]:
                key = normalize_test_name(name)
                if key in index:
                    previous = index[key]
                    if previous.id != document.id:
                        errors.append(
                            f"Conflicting keyword {key!r}: "
                            f"{previous.test_name!r} and {document.test_name!r}"
                        )
                else:
                    index[key] = document
        if errors:
            # The loader accepted the schema but cached an ambiguous snapshot.
            # Discard it so corrected source files can be read on the next try.
            self._loader.clear_cache()
            raise KnowledgeBaseValidationError(errors)
        self._index = index

    def retrieve(self, test_name: str) -> KnowledgeDocument | None:
        key = normalize_test_name(test_name)
        if not key:
            return None
        self._ensure_index()
        assert self._index is not None
        document = self._index.get(key)
        return document.model_copy(deep=True) if document is not None else None
