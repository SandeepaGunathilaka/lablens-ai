"""Text-to-vector encoding only; no retrieval, ranking, or vector persistence."""

import math
from numbers import Real
import os
from pathlib import Path
from typing import Callable, Protocol

from dotenv import load_dotenv

from agents.knowledge_base import KnowledgeDocument


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_TEXT_VERSION = "1"


class Encoder(Protocol):
    def encode(self, sentences: str | list[str], **kwargs): ...


def build_embedding_text(document: KnowledgeDocument) -> str:
    """Build stable text from educational fields without changing the source."""
    def clean(value: str) -> str:
        return " ".join(value.split())

    return "\n".join([
        f"Test: {clean(document.test_name)}",
        "Aliases: " + "; ".join(clean(alias) for alias in document.aliases),
        f"Title: {clean(document.title)}",
        f"Definition: {clean(document.definition)}",
        f"What it measures: {clean(document.what_it_measures)}",
        f"General information: {clean(document.general_information)}",
    ])


def _create_model(model_name: str, **kwargs) -> Encoder:
    # Keep imports, model loading, and potential downloads out of module import.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, **kwargs)


def _as_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _validate_vector(value) -> list[float]:
    value = _as_list(value)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Embedding must be a nonempty one-dimensional numeric vector")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError("Embedding values must be real numbers in one dimension")
        try:
            number = float(item)
        except (OverflowError, ValueError) as exc:
            raise ValueError("Embedding values must be finite") from exc
        if not math.isfinite(number):
            raise ValueError("Embedding values must be finite")
        result.append(number)
    return result


class EmbeddingService:
    """Lazy, reusable encoder with injectable models and ordinary-list outputs."""

    def __init__(
        self, *, model: Encoder | None = None,
        model_factory: Callable[..., Encoder] | None = None,
        model_name: str | None = None, device: str | None = None,
        local_files_only: bool | None = None, revision: str | None = None,
    ):
        if model is not None and model_factory is not None:
            raise ValueError("Provide either model or model_factory, not both")
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        self.model_name = model_name if model_name is not None else os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)
        self.device = device if device is not None else os.getenv("EMBEDDING_DEVICE", "cpu")
        self.revision = (revision if revision is not None else os.getenv("EMBEDDING_MODEL_REVISION", "")).strip() or None
        if not self.model_name.strip() or not self.device.strip():
            raise ValueError("Embedding model name and device must not be blank")
        if local_files_only is None:
            setting = os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "false").strip().lower()
            if setting not in {"true", "false"}:
                raise ValueError("EMBEDDING_LOCAL_FILES_ONLY must be true or false")
            local_files_only = setting == "true"
        self.local_files_only = local_files_only
        self._model = model
        self._model_factory = model_factory if model_factory is not None else _create_model

    def _ensure_model(self) -> Encoder:
        if self._model is None:
            options = {"device": self.device, "local_files_only": self.local_files_only}
            if self.revision is not None:
                options["revision"] = self.revision
            try:
                model = self._model_factory(self.model_name, **options)
                if not callable(getattr(model, "encode", None)):
                    raise TypeError("Model must provide encode()")
            except Exception as exc:
                raise RuntimeError(f"Could not load embedding model {self.model_name!r}: {exc}") from exc
            self._model = model
        return self._model

    def encode_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Embedding text must not be blank")
        output = self._ensure_model().encode(
            text, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )
        return _validate_vector(output)

    def encode_documents(self, documents: list[KnowledgeDocument]) -> list[list[float]]:
        if not documents:
            return []
        texts = [build_embedding_text(document) for document in documents]
        output = _as_list(self._ensure_model().encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        ))
        if not isinstance(output, (list, tuple)) or len(output) != len(documents):
            raise ValueError("Embedding count must match document count")
        vectors = [_validate_vector(vector) for vector in output]
        if len({len(vector) for vector in vectors}) != 1:
            raise ValueError("Embedding vectors must have the same dimension")
        return vectors
